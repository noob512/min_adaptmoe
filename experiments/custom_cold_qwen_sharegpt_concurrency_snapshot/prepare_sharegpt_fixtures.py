#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare disjoint ShareGPT evaluation and heat-calibration fixtures."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--request-count", type=int, default=32)
    parser.add_argument("--prompt-min-tokens", type=int, default=300)
    parser.add_argument("--prompt-max-tokens", type=int, default=320)
    parser.add_argument("--output-tokens", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    source = json.loads(args.dataset.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for source_index, item in enumerate(source):
        conversations = item.get("conversations") if isinstance(item, dict) else None
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        first = conversations[0]
        if not isinstance(first, dict):
            continue
        role = str(first.get("from", "")).strip().lower()
        prompt = first.get("value")
        if role not in {"human", "user"} or not isinstance(prompt, str) or not prompt:
            continue
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if not args.prompt_min_tokens <= prompt_tokens <= args.prompt_max_tokens:
            continue
        selected.append(
            {
                "source_index": source_index,
                "role": role,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
            }
        )
        if len(selected) == 2 * args.request_count:
            break

    if len(selected) != 2 * args.request_count:
        raise SystemExit(
            f"expected {2 * args.request_count} matching requests, found {len(selected)}"
        )

    evaluation = selected[: args.request_count]
    calibration = selected[args.request_count :]
    reference = json.loads(args.reference_snapshot.read_text(encoding="utf-8"))
    expected_indices = reference["source_selection"]["source_indices"]
    actual_indices = [item["source_index"] for item in evaluation]
    if actual_indices != expected_indices:
        raise SystemExit(
            "evaluation fixture does not match the native snapshot: "
            f"expected={expected_indices}, actual={actual_indices}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_fixture(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        requests = [
            {
                "request": index,
                "prompt": item["prompt"],
                "prompt_tokens": item["prompt_tokens"],
                "max_tokens": args.output_tokens,
            }
            for index, item in enumerate(records)
        ]
        metadata = {
            "name": name,
            "selection": "human_first_length_matched_disjoint_split",
            "request_count": len(requests),
            "source_indices": [item["source_index"] for item in records],
            "roles": [item["role"] for item in records],
            "prompt_token_min": min(item["prompt_tokens"] for item in records),
            "prompt_token_max": max(item["prompt_tokens"] for item in records),
            "prompt_token_total": sum(item["prompt_tokens"] for item in records),
            "prompt_token_mean": sum(item["prompt_tokens"] for item in records)
            / len(records),
            "prompt_sha256": sha256_json([item["prompt"] for item in records]),
        }
        fixture = {"metadata": metadata, "requests": requests}
        path = args.output_dir / f"{name}_fixture.json"
        path.write_text(
            json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata["fixture_path"] = str(path)
        metadata["fixture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return metadata

    evaluation_metadata = write_fixture("evaluation", evaluation)
    calibration_metadata = write_fixture("calibration", calibration)
    metadata = {
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": args.model,
        "prompt_min_tokens": args.prompt_min_tokens,
        "prompt_max_tokens": args.prompt_max_tokens,
        "output_tokens": args.output_tokens,
        "evaluation": evaluation_metadata,
        "calibration": calibration_metadata,
        "disjoint_source_indices": not bool(
            set(evaluation_metadata["source_indices"])
            & set(calibration_metadata["source_indices"])
        ),
        "evaluation_matches_native_snapshot": True,
    }
    (args.output_dir / "fixture_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
