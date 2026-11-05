#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--request-count", type=int, default=64)
    parser.add_argument("--prompt-min-tokens", type=int, default=1)
    parser.add_argument("--prompt-max-tokens", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    source = json.loads(args.dataset.read_text(encoding="utf-8"))
    requests = []
    source_indices = []
    prompt_lengths = []
    for source_index, item in enumerate(source):
        conversations = item.get("conversations") if isinstance(item, dict) else None
        if not isinstance(conversations, list) or not conversations:
            continue
        first = conversations[0]
        prompt = first.get("value") if isinstance(first, dict) else None
        if prompt is None and isinstance(first, dict):
            prompt = first.get("content")
        if not isinstance(prompt, str) or not prompt:
            continue
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if not args.prompt_min_tokens <= prompt_tokens <= args.prompt_max_tokens:
            continue
        requests.append(
            {
                "request": len(requests),
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "source_index": source_index,
            }
        )
        source_indices.append(source_index)
        prompt_lengths.append(prompt_tokens)
        if len(requests) == args.request_count:
            break

    if len(requests) != args.request_count:
        raise SystemExit(
            f"expected {args.request_count} requests, found {len(requests)}"
        )

    metadata = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "model": args.model,
        "selection": "first_conversation_prompt_in_token_range",
        "tokenizer_add_special_tokens": False,
        "request_count": len(requests),
        "prompt_min_tokens": min(prompt_lengths),
        "prompt_max_tokens": max(prompt_lengths),
        "prompt_mean_tokens": sum(prompt_lengths) / len(prompt_lengths),
        "prompt_total_tokens": sum(prompt_lengths),
        "source_indices": source_indices,
    }
    payload = {"metadata": metadata, "requests": requests}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["fixture_sha256"] = sha256(args.output)
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
