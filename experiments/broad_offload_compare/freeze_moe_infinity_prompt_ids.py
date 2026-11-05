#!/usr/bin/env python3
"""Freeze one tokenizer-exact DeepSeek prompt without importing MoE-Infinity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SEED_TEXTS = {
    "evaluation": (
        "A careful systems evaluation separates model loading, prompt prefill, "
        "and autoregressive decoding. It records exact token identifiers, memory "
        "budgets, cache behavior, and transfer decisions before comparing latency "
        "or throughput. This evaluation passage is repeated only to create a "
        "deterministic tokenizer-exact prompt. "
    ),
    "calibration": (
        "A calibration workload chooses runtime parameters without observing the "
        "held-out evaluation measurements. It verifies deterministic output, "
        "records every candidate, and freezes the selected configuration before "
        "the formal comparison begins. This calibration passage is repeated only "
        "to create a distinct tokenizer-exact prompt. "
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_sha256(token_ids: list[int]) -> str:
    payload = ",".join(str(value) for value in token_ids).encode("ascii")
    return sha256_bytes(payload)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--role", choices=tuple(SEED_TEXTS), required=True)
    parser.add_argument("--prompt-length", type=int, default=512)
    args = parser.parse_args()
    if args.prompt_length <= 0:
        parser.error("--prompt-length must be positive")

    from transformers import AutoTokenizer, __version__ as transformers_version

    model = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    seed_text = SEED_TEXTS[args.role]
    seed_ids = [
        int(value)
        for value in tokenizer.encode(seed_text, add_special_tokens=False)
    ]
    if not seed_ids:
        raise RuntimeError("seed text tokenized to an empty sequence")
    repeats = (args.prompt_length + len(seed_ids) - 1) // len(seed_ids)
    prompt_ids = (seed_ids * repeats)[: args.prompt_length]
    if len(prompt_ids) != args.prompt_length:
        raise AssertionError("failed to construct the exact prompt length")
    tokenizer_size = len(tokenizer)
    if any(value < 0 or value >= tokenizer_size for value in prompt_ids):
        raise RuntimeError("generated token ID is outside the tokenizer vocabulary")

    tokenizer_assets = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenization_deepseek_fast.py",
        "special_tokens_map.json",
    ):
        digest = sha256_file(model / name)
        if digest is not None:
            tokenizer_assets[name] = digest

    result = {
        "schema_version": 1,
        "fixture_role": args.role,
        "model_snapshot": str(model),
        "model_snapshot_name": model.name,
        "prompt_length": args.prompt_length,
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": token_sha256(prompt_ids),
        "seed_text_sha256": sha256_bytes(seed_text.encode("utf-8")),
        "seed_token_count": len(seed_ids),
        "token_hash_encoding": "ascii comma-separated decimal token IDs",
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "size": tokenizer_size,
            "transformers_version": transformers_version,
            "assets_sha256": tokenizer_assets,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
