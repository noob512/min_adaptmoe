#!/usr/bin/env python3
"""Freeze tokenizer output into a shared cross-framework request fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"missing input fixture: {args.input}")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise SystemExit(f"invalid local model snapshot: {args.model}")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("input and output fixtures must be different files")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise SystemExit("input fixture must be an object with a requests list")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit("input fixture metadata is missing")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=False, local_files_only=True
    )
    frozen_requests: list[dict[str, Any]] = []
    all_ids: list[list[int]] = []
    for index, request in enumerate(payload["requests"]):
        if not isinstance(request, dict) or not isinstance(request.get("prompt"), str):
            raise SystemExit(f"request {index} has no prompt string")
        token_ids = [
            int(value)
            for value in tokenizer.encode(
                request["prompt"], add_special_tokens=False
            )
        ]
        declared = request.get("prompt_tokens")
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise SystemExit(f"request {index} lacks integer prompt_tokens")
        if len(token_ids) != declared:
            raise SystemExit(
                f"request {index} retokenized length {len(token_ids)} != {declared}"
            )
        frozen_requests.append(
            {
                **request,
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": sha256_json(token_ids),
            }
        )
        all_ids.append(token_ids)

    frozen = {
        "metadata": {
            **metadata,
            "prompt_token_ids_frozen": True,
            "tokenizer_add_special_tokens": False,
            "tokenizer_snapshot": str(args.model.resolve()),
            "tokenizer_snapshot_name": args.model.name,
            "tokenizer_config_sha256": (
                sha256_bytes((args.model / "tokenizer_config.json").read_bytes())
                if (args.model / "tokenizer_config.json").is_file()
                else None
            ),
            "source_fixture": str(args.input.resolve()),
            "source_fixture_sha256": sha256_bytes(args.input.read_bytes()),
            "prompt_token_ids_list_sha256": sha256_json(all_ids),
        },
        "requests": frozen_requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_bytes(args.output.read_bytes()),
                "request_count": len(frozen_requests),
                "prompt_token_ids_list_sha256": sha256_json(all_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
