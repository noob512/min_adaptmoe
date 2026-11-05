#!/usr/bin/env python3
"""Validate and summarize selected independent MoE-Infinity qualification cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ALL_MODES = (
    "demand-only",
    "upstream-speculative",
    "fixed-speculative",
    "overlap",
)
EXPECTED_MODE_CONFIG = {
    "demand-only": {
        "speculative_prefetch": False,
        "speculative_prefetch_overlap": False,
        "correction_layer_mode": "upstream",
    },
    "upstream-speculative": {
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": False,
        "correction_layer_mode": "upstream",
    },
    "fixed-speculative": {
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": False,
        "correction_layer_mode": "current",
    },
    "overlap": {
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": True,
        "correction_layer_mode": "current",
    },
}
PAPER_DEEPSEEK_TPOT_MS = 155.0
EXPECTED_MOEI_COMMIT = "ba5651897a80d9c9b7a1500cef2c68adaa63db0f"
EXPECTED_MODEL_SNAPSHOT = "85864749cd611b4353ce1decdb286193298f64c7"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_sidecar(path: Path, failures: list[str]) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not path.is_file() or not sidecar.is_file():
        failures.append(f"missing evidence or SHA-256 sidecar: {path}")
        return
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or fields[0] != sha256_file(path):
        failures.append(f"SHA-256 sidecar mismatch: {path}")


def display_number(value: Any, digits: int = 3, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}{suffix}"
    return "n/a"


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not modes:
        raise argparse.ArgumentTypeError("at least one mode is required")
    if len(modes) != len(set(modes)):
        raise argparse.ArgumentTypeError("modes must not contain duplicates")
    unknown = [mode for mode in modes if mode not in ALL_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown modes: {unknown}")
    return modes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=ALL_MODES,
        help="comma-separated completed cases to summarize",
    )
    parser.add_argument("--expected-prompt-tokens", type=int, default=512)
    parser.add_argument("--expected-output-tokens", type=int, default=32)
    parser.add_argument("--expected-warmup-requests", type=int, default=3)
    parser.add_argument("--expected-repetitions", type=int, default=5)
    parser.add_argument("--expected-num-threads", type=int, default=1)
    parser.add_argument("--expected-prompt-fixture-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    for name, value in (
        ("expected prompt tokens", args.expected_prompt_tokens),
        ("expected output tokens", args.expected_output_tokens),
        ("expected warmup requests", args.expected_warmup_requests),
        ("expected repetitions", args.expected_repetitions),
        ("expected num threads", args.expected_num_threads),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")

    failures: list[str] = []
    try:
        args.run_root.resolve().relative_to(Path("/tmp"))
    except ValueError:
        pass
    else:
        failures.append("run_root is under /tmp; raw evidence is not persistent")

    metadata_root = args.run_root / "metadata"
    required_metadata = (
        "moe_infinity.commit",
        "moe_infinity.status",
        "moe_infinity.patch",
        "model_config.sha256",
        "model_index.sha256",
        "store_extension.sha256",
        "store_build_receipt.json",
        "experiment_scripts.sha256",
        "host_repo.commit",
        "uname.txt",
        "model_shards.tsv",
        "offload_filesystem.txt",
        "prompt_fixture.json",
        "prompt_fixture.source",
        "protocol.env",
        "lscpu.txt",
        "nvidia_smi_q.txt",
        "pip_freeze.txt",
    )
    for name in required_metadata:
        if not (metadata_root / name).is_file():
            failures.append(f"missing metadata evidence: {name}")
    commit_path = metadata_root / "moe_infinity.commit"
    if commit_path.is_file():
        commit = commit_path.read_text(encoding="utf-8").strip()
        if commit != EXPECTED_MOEI_COMMIT:
            failures.append(f"MoE-Infinity commit mismatch: {commit!r}")
    verify_sha256_sidecar(metadata_root / "moe_infinity.patch", failures)
    verify_sha256_sidecar(metadata_root / "model_shards.tsv", failures)
    verify_sha256_sidecar(metadata_root / "prompt_fixture.json", failures)
    verify_sha256_sidecar(metadata_root / "store_build_receipt.json", failures)
    frozen_prompt_ids: list[int] | None = None
    prompt_fixture_path = metadata_root / "prompt_fixture.json"
    if prompt_fixture_path.is_file():
        if sha256_file(prompt_fixture_path) != args.expected_prompt_fixture_sha256:
            failures.append("frozen prompt fixture SHA-256 mismatch")
        prompt_fixture = load_json(prompt_fixture_path)
        candidate_ids = prompt_fixture.get("prompt_token_ids")
        if isinstance(candidate_ids, list) and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in candidate_ids
        ):
            frozen_prompt_ids = candidate_ids
        else:
            failures.append("frozen prompt fixture has invalid prompt_token_ids")

    if args.modes == ("demand-only",):
        for name in ("offload_before.tsv", "offload_after.tsv"):
            verify_sha256_sidecar(metadata_root / name, failures)
        after_inventory = metadata_root / "offload_after.tsv"
        if after_inventory.is_file():
            inventory_text = after_inventory.read_text(encoding="utf-8")
            if "\narcher_index\t" not in inventory_text:
                failures.append("offload_after.tsv does not contain archer_index")
            if "\narcher_param_" not in inventory_text:
                failures.append("offload_after.tsv does not contain archer_param data")
    cases: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    output_hashes: set[str] = set()
    model_names: set[str] = set()
    offload_paths: set[str] = set()
    for mode in args.modes:
        case_root = args.run_root / "cases" / mode
        result_path = case_root / "result.json"
        returncode_path = case_root / "returncode.txt"
        if not result_path.is_file() or not returncode_path.is_file():
            failures.append(f"{mode}: missing result.json or returncode.txt")
            continue
        for name in (
            "command.txt",
            "environment.txt",
            "stdout.log",
            "stderr.log",
            "nvml.csv",
            "nvml.err",
        ):
            if not (case_root / name).is_file():
                failures.append(f"{mode}: missing raw case evidence {name}")
        result = load_json(result_path)
        returncode = returncode_path.read_text(encoding="utf-8").strip()
        if returncode != "0":
            failures.append(f"{mode}: returncode={returncode}")
        if result.get("status") != "ok":
            failures.append(f"{mode}: status={result.get('status')!r}")
        qualification = result.get("qualification", {})
        if qualification.get("status") != "ok":
            failures.append(f"{mode}: qualification checks failed")
        if result.get("mode") != mode:
            failures.append(f"{mode}: result mode={result.get('mode')!r}")
        if result.get("mode_config") != EXPECTED_MODE_CONFIG[mode]:
            failures.append(f"{mode}: mode_config does not match the selected mode")
        protocol = result.get("protocol", {})
        if protocol.get("batch_size") != 1:
            failures.append(f"{mode}: batch_size is not 1")
        if protocol.get("prompt_tokens") != args.expected_prompt_tokens:
            failures.append(
                f"{mode}: prompt length is not {args.expected_prompt_tokens}"
            )
        if protocol.get("output_tokens") != args.expected_output_tokens:
            failures.append(
                f"{mode}: output length is not {args.expected_output_tokens}"
            )
        if protocol.get("warmup_requests") != args.expected_warmup_requests:
            failures.append(
                f"{mode}: warmup_requests is not {args.expected_warmup_requests}"
            )
        if protocol.get("measured_repetitions") != args.expected_repetitions:
            failures.append(
                f"{mode}: measured_repetitions is not {args.expected_repetitions}"
            )
        if protocol.get("expert_compute_threads") != args.expected_num_threads:
            failures.append(
                f"{mode}: expert_compute_threads is not {args.expected_num_threads}"
            )
        if protocol.get("greedy") is not True:
            failures.append(f"{mode}: greedy is not true")
        if protocol.get("use_native_engine") is not False:
            failures.append(f"{mode}: use_native_engine is not false")

        warmups = result.get("warmups")
        trials = result.get("trials")
        if not isinstance(warmups, list):
            failures.append(f"{mode}: warmups is not a list")
            warmups = []
        if not isinstance(trials, list):
            failures.append(f"{mode}: trials is not a list")
            trials = []
        if len(warmups) != args.expected_warmup_requests:
            failures.append(
                f"{mode}: recorded {len(warmups)} warmups, expected "
                f"{args.expected_warmup_requests}"
            )
        if len(trials) != args.expected_repetitions:
            failures.append(
                f"{mode}: recorded {len(trials)} trials, expected "
                f"{args.expected_repetitions}"
            )
        for expected_warmup, records in ((True, warmups), (False, trials)):
            for expected_index, record in enumerate(records):
                if record.get("warmup") is not expected_warmup:
                    failures.append(
                        f"{mode}: record {expected_index} has wrong warmup flag"
                    )
                if record.get("trial_index") != expected_index:
                    failures.append(
                        f"{mode}: record {expected_index} has wrong trial_index"
                    )
                if record.get("prompt_tokens") != args.expected_prompt_tokens:
                    failures.append(
                        f"{mode}: record {expected_index} has wrong prompt length"
                    )
                if record.get("output_tokens") != args.expected_output_tokens:
                    failures.append(
                        f"{mode}: record {expected_index} has wrong output length"
                    )

        resolved = result.get("resolved_archer_config", {})
        expected_config = EXPECTED_MODE_CONFIG[mode]
        if resolved.get("num_threads") != args.expected_num_threads:
            failures.append(f"{mode}: resolved num_threads mismatch")
        if resolved.get("device_memory_ratio") != protocol.get(
            "device_memory_ratio"
        ):
            failures.append(f"{mode}: resolved device_memory_ratio mismatch")
        if resolved.get("use_native_engine") is not False:
            failures.append(f"{mode}: resolved use_native_engine is not false")
        if resolved.get("prefetch") is not False:
            failures.append(f"{mode}: legacy prefetch unexpectedly enabled")
        if (
            resolved.get("speculative_prefetch")
            is not expected_config["speculative_prefetch"]
        ):
            failures.append(f"{mode}: resolved speculative_prefetch mismatch")
        if (
            resolved.get("speculative_prefetch_overlap")
            is not expected_config["speculative_prefetch_overlap"]
        ):
            failures.append(f"{mode}: resolved overlap mismatch")

        offload_path = result.get("offload_dir")
        if not isinstance(offload_path, str):
            failures.append(f"{mode}: offload_dir missing")
        else:
            offload_paths.add(offload_path)
            if resolved.get("offload_path") != str(Path(offload_path).resolve()):
                failures.append(f"{mode}: resolved offload_path mismatch")
            try:
                Path(offload_path).resolve().relative_to(Path("/tmp"))
            except ValueError:
                pass
            else:
                failures.append(f"{mode}: offload_dir is under /tmp")
        prompt_hash = result.get("prompt", {}).get("prompt_token_ids_sha256")
        if isinstance(prompt_hash, str):
            prompt_hashes.add(prompt_hash)
        else:
            failures.append(f"{mode}: prompt token hash missing")
        result_prompt = result.get("prompt", {})
        if (
            result_prompt.get("source_file_sha256")
            != args.expected_prompt_fixture_sha256
        ):
            failures.append(f"{mode}: frozen prompt source SHA-256 mismatch")
        if frozen_prompt_ids is None or result_prompt.get(
            "prompt_token_ids"
        ) != frozen_prompt_ids:
            failures.append(f"{mode}: runtime prompt IDs differ from frozen fixture")
        case_output_hashes = result.get("summary", {}).get("output_hashes", [])
        if len(case_output_hashes) != 1:
            failures.append(f"{mode}: output hash is not stable")
        else:
            output_hashes.add(case_output_hashes[0])
        model_names.add(str(result.get("model")))
        if Path(str(result.get("model"))).name != EXPECTED_MODEL_SNAPSHOT:
            failures.append(f"{mode}: model snapshot mismatch")

        environment = result.get("environment", {})
        if environment.get("omp_num_threads") != str(args.expected_num_threads):
            failures.append(f"{mode}: OMP_NUM_THREADS evidence mismatch")
        if environment.get("moe_infinity_commit") != EXPECTED_MOEI_COMMIT:
            failures.append(f"{mode}: runtime MoE-Infinity commit mismatch")
        metadata_patch = metadata_root / "moe_infinity.patch"
        if metadata_patch.is_file() and environment.get(
            "moe_infinity_diff_sha256"
        ) != sha256_file(metadata_patch):
            failures.append(f"{mode}: runtime/vendor patch SHA-256 mismatch")
        status_path = metadata_root / "moe_infinity.status"
        if status_path.is_file() and environment.get(
            "moe_infinity_status_porcelain"
        ) != status_path.read_text(encoding="utf-8").strip():
            failures.append(f"{mode}: runtime/vendor dirty status mismatch")
        extension_manifest = metadata_root / "store_extension.sha256"
        if extension_manifest.is_file():
            fields = extension_manifest.read_text(encoding="utf-8").split()
            if not fields or environment.get("store_extension_sha256") != fields[0]:
                failures.append(f"{mode}: runtime extension SHA-256 mismatch")

        all_request_hashes = [
            record.get("generated_token_ids_sha256")
            for record in [*warmups, *trials]
        ]
        if (
            not all_request_hashes
            or not all(isinstance(value, str) for value in all_request_hashes)
            or len(set(all_request_hashes)) != 1
        ):
            failures.append(f"{mode}: warmup/measured output hashes differ")
        if qualification.get("warmup_and_measured_output_hash_stable") is not True:
            failures.append(f"{mode}: lifecycle output-hash gate did not pass")
        final_counters = qualification.get("instrumentation_final", {})
        if qualification.get("recorded_warmups") != args.expected_warmup_requests:
            failures.append(f"{mode}: qualification warmup count mismatch")
        if qualification.get("recorded_repetitions") != args.expected_repetitions:
            failures.append(f"{mode}: qualification repetition count mismatch")
        if mode == "demand-only":
            unexpected = {
                field: final_counters.get(field)
                for field in (
                    "prediction_calls",
                    "correction_calls",
                    "enqueue_attempts",
                    "enqueue_errors",
                )
                if final_counters.get(field) != 0
            }
            if unexpected:
                failures.append(
                    f"{mode}: lifecycle prefetch counters are not zero: {unexpected}"
                )

        tpot = result.get("summary", {}).get("decode_tpot_seconds", {})
        ttft = result.get("summary", {}).get("ttft_seconds", {})
        e2e = result.get("summary", {}).get("e2e_seconds", {})
        counters = qualification.get("instrumentation_final", {})
        median_tpot_ms = (
            float(tpot["median"]) * 1000.0 if tpot.get("median") is not None else None
        )
        cases.append(
            {
                "mode": mode,
                "result_path": str(result_path.resolve()),
                "load_seconds": result.get("load_seconds"),
                "median_ttft_ms": (
                    float(ttft["median"]) * 1000.0
                    if ttft.get("median") is not None
                    else None
                ),
                "mean_ttft_ms": (
                    float(ttft["mean"]) * 1000.0
                    if ttft.get("mean") is not None
                    else None
                ),
                "median_decode_tpot_ms": median_tpot_ms,
                "mean_decode_tpot_ms": (
                    float(tpot["mean"]) * 1000.0
                    if tpot.get("mean") is not None
                    else None
                ),
                "median_e2e_ms": (
                    float(e2e["median"]) * 1000.0
                    if e2e.get("median") is not None
                    else None
                ),
                "paper_anchor_tpot_ratio": (
                    median_tpot_ms / PAPER_DEEPSEEK_TPOT_MS
                    if median_tpot_ms is not None
                    else None
                ),
                "prediction_calls_total": counters.get("prediction_calls"),
                "correction_calls_total": counters.get("correction_calls"),
                "enqueue_attempts_total": counters.get("enqueue_attempts"),
                "enqueue_errors_total": counters.get("enqueue_errors"),
                "requested_effective_layer_differences_total": counters.get(
                    "requested_effective_layer_differences"
                ),
                "output_hash": (
                    case_output_hashes[0] if len(case_output_hashes) == 1 else None
                ),
                "qualification": qualification,
            }
        )

    if len(prompt_hashes) != 1:
        failures.append(f"profiles used {len(prompt_hashes)} distinct prompt hashes")
    if len(output_hashes) != 1:
        failures.append(f"profiles produced {len(output_hashes)} distinct output hashes")
    if len(model_names) != 1:
        failures.append(f"profiles used different models: {sorted(model_names)}")
    if len(offload_paths) != 1:
        failures.append(f"profiles used different offload paths: {sorted(offload_paths)}")
    if len(cases) != len(args.modes):
        failures.append(
            f"found {len(cases)} complete cases, expected {len(args.modes)}"
        )

    public_scope = (
        "public on-demand expert-cache path; not the paper EAM/EAMC implementation"
        if args.modes == ("demand-only",)
        else (
            "public on-demand path plus router-logit top-2 speculative diagnostics "
            "with optional layer-index patch; not the paper EAM/EAMC implementation"
        )
    )
    snapshot = {
        "schema_version": 1,
        "experiment": "moe_infinity_public_artifact_qualification",
        "status": "ok" if not failures else "error",
        "run_root": str(args.run_root.resolve()),
        "modes": list(args.modes),
        "expected_protocol": {
            "batch_size": 1,
            "prompt_tokens": args.expected_prompt_tokens,
            "output_tokens": args.expected_output_tokens,
            "warmup_requests": args.expected_warmup_requests,
            "measured_repetitions": args.expected_repetitions,
            "expert_compute_threads": args.expected_num_threads,
            "prompt_fixture_sha256": args.expected_prompt_fixture_sha256,
        },
        "paper_anchor": {
            "model": "DeepSeek-V2-Lite",
            "gpu": "NVIDIA RTX A5000 24GB",
            "pcie": "PCIe 4.0, paper reports 24 GB/s in Table 1",
            "prompt_tokens": 512,
            "output_tokens": 32,
            "reported_average_tpot_ms": PAPER_DEEPSEEK_TPOT_MS,
            "comparability": (
                "protocol anchor only: this machine, public artifact, software stack, "
                "and GPU differ from the paper"
            ),
        },
        "public_artifact_scope": public_scope,
        "prompt_token_ids_sha256": next(iter(prompt_hashes), None),
        "cross_mode_output_hash_equal": len(output_hashes) == 1,
        "offload_dir": next(iter(offload_paths), None),
        "cases": cases,
        "failures": failures,
    }
    atomic_text(
        args.output_json,
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    rows: list[str] = []
    for case in cases:
        rows.append(
            "| {mode} | {ttft} | {tpot} | {e2e} | {pred} | "
            "{corr} | {enqueue} | {ratio} |".format(
                mode=case["mode"],
                ttft=display_number(case["median_ttft_ms"]),
                tpot=display_number(case["median_decode_tpot_ms"]),
                e2e=display_number(case["median_e2e_ms"]),
                pred=case["prediction_calls_total"],
                corr=case["correction_calls_total"],
                enqueue=case["enqueue_attempts_total"],
                ratio=display_number(case["paper_anchor_tpot_ratio"], suffix="x"),
            )
        )
    markdown = "\n".join(
        [
            "# MoE-Infinity public-artifact qualification result",
            "",
            f"Status: `{snapshot['status']}`",
            "",
            f"These values use batch=1, an exact {args.expected_prompt_tokens}-token "
            f"prompt, {args.expected_output_tokens} greedy output tokens, "
            f"{args.expected_warmup_requests} warmups, "
            f"{args.expected_repetitions} measured repetitions, and independent "
            f"cold processes. Scope: {public_scope}.",
            "",
            "| Mode | Median TTFT ms | Median decode TPOT ms | Median E2E ms | "
            "Prediction calls | Correction calls | Enqueues | TPOT / paper 155 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The final column is only a protocol anchor. The paper used an RTX "
            "A5000 and a different extreme-performance implementation, so it is "
            "not an apples-to-apples speedup or regression.",
            "The one-thread demand qualification is a stability point, not the "
            "current public artifact's four-thread vendor-best setting.",
            "",
            "## Integrity",
            "",
            f"- Cross-mode output hash equal: `{snapshot['cross_mode_output_hash_equal']}`",
            f"- Prompt token hash: `{snapshot['prompt_token_ids_sha256']}`",
            f"- Failures: `{failures}`",
            "",
        ]
    )
    atomic_text(args.output_md, markdown)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
