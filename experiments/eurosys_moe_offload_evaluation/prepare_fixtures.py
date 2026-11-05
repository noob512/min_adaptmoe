#!/usr/bin/env python3
"""Freeze disjoint, pre-tokenized EuroSys evaluation fixtures.

This program intentionally consumes only the source dataset and tokenizer.  It
does not import CUDA, run a model, inspect routing, or read any old result.
Every source conversation is assigned to at most one split/workload pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_DATASET_SHA256 = (
    "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
)
DEFAULT_MODEL = Path(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/"
    "85864749cd611b4353ce1decdb286193298f64c7"
)
DEFAULT_DATASET = Path(
    "/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/datasets/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
SPLITS = ("calibration", "validation", "evaluation", "shift")
WORKLOADS = {
    "short_dialogue": {"prompt_tokens": 128, "output_tokens": 32},
    "ordinary_chat": {"prompt_tokens": 512, "output_tokens": 128},
    "medium_context": {"prompt_tokens": 2048, "output_tokens": 128},
    "rag_summary": {"prompt_tokens": 8192, "output_tokens": 128},
    "long_generation": {"prompt_tokens": 512, "output_tokens": 512},
}
CODE_RE = re.compile(
    r"```|\b(?:python|javascript|typescript|java|c\+\+|rust|golang|sql)\b|"
    r"\b(?:function|class|def|import|compile|debug|algorithm|代码|编程)\b",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_sha256(values: list[int]) -> str:
    encoded = ",".join(map(str, values)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stable_bucket(source_index: int, seed: int, modulo: int) -> int:
    payload = f"{seed}:{source_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def first_user_text(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    conversations = value.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    first = conversations[0]
    if not isinstance(first, dict):
        return None
    role = str(first.get("from", "")).strip().lower()
    text = first.get("value")
    if role not in {"human", "user"} or not isinstance(text, str):
        return None
    text = text.strip()
    return text or None


@dataclass(frozen=True)
class SourcePrompt:
    source_index: int
    token_ids: tuple[int, ...]
    code_like: bool


def pack_requests(
    sources: list[SourcePrompt],
    *,
    count: int,
    prompt_tokens: int,
    separator_id: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    requests: list[dict[str, Any]] = []
    used: set[int] = set()
    cursor = 0
    while len(requests) < count:
        packed: list[int] = []
        indices: list[int] = []
        original_lengths: list[int] = []
        while len(packed) < prompt_tokens:
            if cursor >= len(sources):
                raise RuntimeError(
                    f"insufficient source prompts: made {len(requests)}/{count} "
                    f"requests of {prompt_tokens} tokens"
                )
            source = sources[cursor]
            cursor += 1
            if source.source_index in used:
                raise AssertionError("source index was reused")
            if packed:
                packed.append(separator_id)
            remaining = prompt_tokens - len(packed)
            packed.extend(source.token_ids[:remaining])
            indices.append(source.source_index)
            original_lengths.append(len(source.token_ids))
            used.add(source.source_index)
        if len(packed) != prompt_tokens:
            raise AssertionError("packed request length drift")
        requests.append(
            {
                "request": len(requests),
                "prompt": "",
                "prompt_tokens": prompt_tokens,
                "prompt_token_ids": packed,
                "prompt_token_ids_sha256": token_sha256(packed),
                "source_indices": indices,
                "source_original_token_lengths": original_lengths,
                "construction": "ordered_pack_with_eos_separator_and_final_truncation",
            }
        )
    return requests, used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--requests-per-workload", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{5,127}", args.protocol_id):
        raise SystemExit("invalid --protocol-id")
    if args.requests_per_workload < 8:
        raise SystemExit("at least eight requests per workload are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    dataset_sha = sha256_file(args.dataset)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise SystemExit(f"dataset SHA-256 drift: {dataset_sha}")
    config = args.model / "config.json"
    index = args.model / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise SystemExit("model config/index is incomplete")

    # This is the only heavyweight import.  Offline mode prevents hidden data
    # dependencies from changing after protocol registration.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True
    )
    separator_id = tokenizer.eos_token_id
    if not isinstance(separator_id, int) or separator_id < 0:
        raise SystemExit("tokenizer has no non-negative EOS token ID")

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("ShareGPT source is not a JSON array")
    by_split: dict[str, list[SourcePrompt]] = {name: [] for name in SPLITS}
    non_shift = SPLITS[:3]
    for source_index, item in enumerate(raw):
        text = first_user_text(item)
        if text is None:
            continue
        code_like = bool(CODE_RE.search(text))
        if code_like:
            split = "shift"
        else:
            split = non_shift[stable_bucket(source_index, args.seed, len(non_shift))]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in token_ids):
            raise RuntimeError("tokenizer returned an invalid token ID")
        by_split[split].append(
            SourcePrompt(source_index, tuple(token_ids), code_like)
        )
    del raw
    rng = random.Random(args.seed)
    for split in SPLITS:
        rng.shuffle(by_split[split])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_used: dict[int, tuple[str, str]] = {}
    file_manifest: dict[str, Any] = {}
    source_counts = {split: len(values) for split, values in by_split.items()}
    for split in SPLITS:
        # Each workload receives its own disjoint slice of the shuffled split.
        cursor = 0
        for workload, spec in WORKLOADS.items():
            remaining = by_split[split][cursor:]
            requests, used = pack_requests(
                remaining,
                count=args.requests_per_workload,
                prompt_tokens=int(spec["prompt_tokens"]),
                separator_id=separator_id,
            )
            consumed = 0
            for source in remaining:
                consumed += 1
                if source.source_index in used and len(used) == consumed:
                    # pack_requests consumes a prefix; this condition is true at
                    # the last element of that prefix even when token truncation
                    # occurs within the source.
                    break
            if set(source.source_index for source in remaining[:consumed]) != used:
                raise AssertionError("packer did not consume an exact source prefix")
            cursor += consumed
            for source_index in used:
                previous = global_used.setdefault(source_index, (split, workload))
                if previous != (split, workload):
                    raise AssertionError("cross-fixture source leakage")
            fixture = {
                "metadata": {
                    "schema_version": 1,
                    "protocol_id": args.protocol_id,
                    "fixture_role": split,
                    "workload": workload,
                    "selection_frozen_before_measurement": True,
                    "source_dataset_sha256": dataset_sha,
                    "source_domain": (
                        "ShareGPT_code_heuristic_shift"
                        if split == "shift"
                        else "ShareGPT_non_code"
                    ),
                    "requests": len(requests),
                    "prompt_tokens": int(spec["prompt_tokens"]),
                    "output_tokens": int(spec["output_tokens"]),
                    "frozen_prompt_token_ids": True,
                    "source_indices": sorted(used),
                    "source_indices_sha256": token_sha256(sorted(used)),
                    "model_snapshot": str(args.model.resolve()),
                    "model_config_sha256": sha256_file(config),
                    "tokenizer_eos_token_id": separator_id,
                },
                "requests": requests,
            }
            path = args.output_dir / f"{split}_{workload}.json"
            atomic_json(path, fixture)
            file_manifest[f"{split}/{workload}"] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "source_count": len(used),
                "prompt_tokens": int(spec["prompt_tokens"]),
                "output_tokens": int(spec["output_tokens"]),
            }

    all_sets = [set(value["metadata"]["source_indices"]) for value in (
        json.loads((args.output_dir / item["path"]).read_text(encoding="utf-8"))
        for item in file_manifest.values()
    )]
    disjoint = all(not left & right for i, left in enumerate(all_sets) for right in all_sets[i + 1 :])
    if not disjoint:
        raise AssertionError("fixture source-index sets overlap")
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "protocol_id": args.protocol_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_is_cpu_only": True,
        "calibration_validation_evaluation_shift_disjoint": True,
        "all_fixture_source_sets_disjoint": disjoint,
        "seed": args.seed,
        "requests_per_workload": args.requests_per_workload,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(config),
        "model_index_sha256": sha256_file(index),
        "tokenizer_class": type(tokenizer).__name__,
        "source_candidate_counts": source_counts,
        "files": file_manifest,
    }
    atomic_json(args.output_dir / "fixture_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
