#!/usr/bin/env python3
"""Extend the original frozen DeepSeek fixture without repeating source requests."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "experiments/eurosys_moe_offload_evaluation"
sys.path.insert(0, str(EVAL))
from prepare_fixtures import (  # noqa: E402
    CODE_RE, DEFAULT_DATASET, DEFAULT_MODEL, EXPECTED_DATASET_SHA256,
    SPLITS, SourcePrompt, atomic_json, first_user_text, pack_requests,
    sha256_file, stable_bucket, token_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    fixture_dir = EVAL / "inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16"
    base_path = fixture_dir / "evaluation_ordinary_chat.json"
    base = json.loads(base_path.read_text())
    old_manifest = json.loads((fixture_dir / "fixture_manifest.json").read_text())
    requests = copy.deepcopy(base["requests"])
    if args.requests <= len(requests):
        raise SystemExit("extension count must exceed the original request count")
    dataset_sha = sha256_file(DEFAULT_DATASET)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise SystemExit("ShareGPT source hash differs")
    used_before: set[int] = set()
    excluded_files = []
    for path in sorted(fixture_dir.glob("*.json")):
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("requests"), list):
            continue
        excluded_files.append({"path": str(path), "sha256": sha256_file(path)})
        used_before.update(index for record in value["requests"] for index in record["source_indices"])

    raw = json.loads(DEFAULT_DATASET.read_text())
    eligible = []
    split_seed = int(old_manifest["seed"])
    eval_bucket = SPLITS[:3].index("evaluation")
    for index, record in enumerate(raw):
        if index in used_before or stable_bucket(index, split_seed, 3) != eval_bucket:
            continue
        text = first_user_text(record)
        if text is not None and not CODE_RE.search(text):
            eligible.append((index, text))
    del raw
    eligible.sort(key=lambda item: hashlib.sha256(f"{args.seed}:{item[0]}".encode("ascii")).digest())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL), trust_remote_code=True, local_files_only=True)
    separator = tokenizer.eos_token_id
    assert separator == base["metadata"]["tokenizer_eos_token_id"]
    seen_hashes = {record["prompt_token_ids_sha256"] for record in requests}
    assert len(seen_hashes) == len(requests)
    sources: list[SourcePrompt] = []
    packed_length = 0
    duplicate_prompts = 0
    for index, text in eligible:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue
        packed_length += len(tokens) + int(bool(sources))
        sources.append(SourcePrompt(index, tuple(tokens), False))
        if packed_length < 512:
            continue
        packed, _ = pack_requests(sources, count=1, prompt_tokens=512, separator_id=separator)
        sources = []
        packed_length = 0
        record = packed[0]
        if record["prompt_token_ids_sha256"] in seen_hashes:
            duplicate_prompts += 1
            continue
        record["request"] = len(requests)
        requests.append(record)
        seen_hashes.add(record["prompt_token_ids_sha256"])
        if len(requests) == args.requests:
            break
    if len(requests) != args.requests:
        raise SystemExit("not enough unused ShareGPT evaluation conversations")

    flat = [index for record in requests for index in record["source_indices"]]
    added = [index for record in requests[len(base["requests"]):] for index in record["source_indices"]]
    assert len(flat) == len(set(flat)) and not set(added) & used_before
    assert all(record["prompt_tokens"] == len(record["prompt_token_ids"]) == 512 for record in requests)
    assert all(token_sha256(record["prompt_token_ids"]) == record["prompt_token_ids_sha256"] for record in requests)
    assert requests[:len(base["requests"])] == base["requests"]
    metadata = copy.deepcopy(base["metadata"])
    metadata.update({
        "protocol_id": "deepseek-p512-n128-qps-scan-20261023",
        "requests": len(requests), "output_tokens": 16,
        "source_indices": sorted(set(flat)),
        "source_indices_sha256": token_sha256(sorted(set(flat))),
        "base_fixture": str(base_path), "base_fixture_sha256": sha256_file(base_path),
        "base_prefix_requests": len(base["requests"]),
        "previous_experiment_prefix_requests": 16,
        "extension_seed": args.seed, "evaluation_split_seed": split_seed,
        "extension_selection": "unused non-code evaluation conversations ordered by SHA256(seed:index)",
        "extension_source_count": len(added),
        "extension_disjoint_from_all_original_fixture_sources": True,
        "duplicate_prompt_hashes": 0, "skipped_duplicate_packed_prompts": duplicate_prompts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_sha256": sha256_file(Path(__file__)),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {"metadata": metadata, "requests": requests})
    manifest = {
        "status": "ok", "generation_is_cpu_only": True,
        "fixture": str(args.output.resolve()), "fixture_sha256": sha256_file(args.output),
        "dataset": str(DEFAULT_DATASET), "dataset_sha256": dataset_sha,
        "requests": len(requests), "prompt_tokens": 512, "output_tokens": 16,
        "base_prefix_requests": len(base["requests"]), "eligible_conversations": len(eligible),
        "total_source_conversations": len(flat), "added_source_conversations": len(added),
        "excluded_fixture_files": excluded_files,
        "original_prefix_preserved": True, "unique_prompt_hashes": len(seen_hashes),
        "unique_source_indices": len(set(flat)), "new_sources_disjoint_from_original_fixtures": True,
    }
    atomic_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "excluded_fixture_files"}, indent=2))


if __name__ == "__main__":
    main()
