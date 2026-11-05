#!/usr/bin/env python3
"""Build a deterministic ShareGPT open-loop trace for offload comparisons.

ShareGPT supplies real request contents but no production arrival timestamps.
This tool therefore freezes the request contents and synthesizes only the
arrival process.  The low/medium/high files contain the same requests in the
same order; only their absolute arrival times differ.

The generated JSONL is deliberately adapter-neutral.  An HTTP adapter can
use ``messages``/``max_tokens`` while a direct vLLM adapter can use the frozen
``prompt_token_ids``.  Both paths can audit that they received identical
token IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INPUT_BINS: tuple[tuple[str, int, int], ...] = (
    ("32-255", 32, 255),
    ("256-1023", 256, 1023),
    ("1024-4095", 1024, 4095),
    ("4096-8192", 4096, 8192),
)
OUTPUT_BINS: tuple[tuple[str, int, int], ...] = (
    ("16-63", 16, 63),
    ("64-127", 64, 127),
    ("128-255", 128, 255),
    ("256-512", 256, 512),
)

ASSISTANT_ROLES = {"gpt", "assistant", "chatgpt", "bing", "bard"}
USER_ROLES = {"human", "user"}
DEFAULT_RATES = {"low": 0.25, "medium": 0.50, "high": 0.75}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.replace("\x00", "").strip()
    if not value or any(ord(char) < 9 for char in value):
        return None
    return value


def bin_name(value: int, bins: tuple[tuple[str, int, int], ...]) -> str | None:
    for name, lower, upper in bins:
        if lower <= value <= upper:
            return name
    return None


def parse_rates(value: str) -> dict[str, float]:
    rates = dict(DEFAULT_RATES)
    if value:
        for item in value.split(","):
            name, separator, raw_rate = item.partition("=")
            if not separator or name not in rates:
                raise ValueError("rates must use low=R,medium=R,high=R")
            rate = float(raw_rate)
            if rate <= 0:
                raise ValueError("all arrival rates must be positive")
            rates[name] = rate
    return rates


def tokenizer_hash(model_path: Path) -> str:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )
    digest = hashlib.sha256()
    found = False
    for name in names:
        path = model_path / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    if not found:
        raise ValueError(f"no tokenizer files found under {model_path}")
    return digest.hexdigest()


def load_candidates(dataset: Path, tokenizer: Any) -> tuple[list[dict[str, Any]], Counter]:
    source = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("ShareGPT source must be a JSON array")

    candidates: list[dict[str, Any]] = []
    rejection_counts: Counter = Counter()
    seen_content: set[str] = set()
    for source_index, item in enumerate(source):
        if not isinstance(item, dict):
            rejection_counts["not_object"] += 1
            continue
        conversations = item.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            rejection_counts["less_than_two_turns"] += 1
            continue
        first, second = conversations[0], conversations[1]
        if not isinstance(first, dict) or not isinstance(second, dict):
            rejection_counts["invalid_turn"] += 1
            continue
        first_role = str(first.get("from", "")).strip().lower()
        second_role = str(second.get("from", "")).strip().lower()
        if first_role not in USER_ROLES or second_role not in ASSISTANT_ROLES:
            rejection_counts["unexpected_roles"] += 1
            continue
        prompt = normalize_text(first.get("value"))
        answer = normalize_text(second.get("value"))
        if prompt is None or answer is None:
            rejection_counts["empty_or_control_text"] += 1
            continue

        content_digest = sha256_bytes((prompt + "\0" + answer).encode("utf-8"))
        if content_digest in seen_content:
            rejection_counts["duplicate_first_turn"] += 1
            continue
        seen_content.add(content_digest)

        prompt_token_ids = [
            int(token)
            for token in tokenizer.encode(prompt, add_special_tokens=False)
        ]
        answer_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
        prompt_tokens = len(prompt_token_ids)
        max_tokens = min(max(answer_tokens, 16), 512)
        input_bin = bin_name(prompt_tokens, INPUT_BINS)
        output_bin = bin_name(max_tokens, OUTPUT_BINS)
        if input_bin is None:
            rejection_counts["prompt_length"] += 1
            continue
        if answer_tokens < 16:
            rejection_counts["output_too_short"] += 1
            continue
        if output_bin is None:
            rejection_counts["output_length"] += 1
            continue
        if prompt_tokens + max_tokens > 8192:
            rejection_counts["prompt_plus_output_length"] += 1
            continue

        candidates.append(
            {
                "source_index": source_index,
                "source_id": item.get("id"),
                "source_content_sha256": content_digest,
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "prompt_tokens": prompt_tokens,
                "reference_assistant_tokens": answer_tokens,
                "max_tokens": max_tokens,
                "input_bin": input_bin,
                "output_bin": output_bin,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
    return candidates, rejection_counts


def select_candidates(
    candidates: list[dict[str, Any]], request_count: int, seed: int
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        cells[(candidate["input_bin"], candidate["output_bin"])].append(candidate)
    expected_cells = len(INPUT_BINS) * len(OUTPUT_BINS)
    if request_count % expected_cells != 0:
        raise ValueError(
            f"request-count must be divisible by {expected_cells} for equal 4x4 quotas"
        )
    quota = request_count // expected_cells
    selector = random.Random(seed)
    selected: list[dict[str, Any]] = []
    cell_manifest: dict[str, Any] = {}
    for input_name, _, _ in INPUT_BINS:
        for output_name, _, _ in OUTPUT_BINS:
            key = (input_name, output_name)
            pool = list(cells.get(key, []))
            selector.shuffle(pool)
            if len(pool) < quota:
                raise ValueError(
                    f"cell {input_name}/{output_name} has {len(pool)} candidates, "
                    f"needs {quota}"
                )
            chosen = pool[:quota]
            selected.extend(chosen)
            cell_manifest[f"{input_name}/{output_name}"] = {
                "candidate_count": len(pool),
                "selected_count": len(chosen),
                "source_indices": [item["source_index"] for item in chosen],
            }

    # A second deterministic shuffle prevents the 4x4 quota blocks from
    # creating an artificial phase pattern in the arrival trace.
    selector.shuffle(selected)
    for request_index, item in enumerate(selected):
        item["request_id"] = f"sharegpt-{request_index:04d}"
    return selected


def build_normalized_arrivals(count: int, seed: int) -> list[float]:
    """Return cumulative inter-arrival times in seconds at 1 req/s."""
    if count <= 0:
        return []
    rng = random.Random(seed)
    # Gamma(k=0.5) retains burstiness while keeping a known mean interval.
    shape = 0.5
    scale = 1.0 / shape
    arrivals_s = [0.0]
    for _ in range(1, count):
        arrivals_s.append(arrivals_s[-1] + rng.gammavariate(shape, scale))
    return arrivals_s


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    # The sparse 4096-8192/16-63 ShareGPT cell has only eight eligible
    # records in the frozen source, so 128 is the largest no-duplication
    # equal-quota trace (8 requests in each 4x4 cell).
    parser.add_argument("--request-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument(
        "--rates",
        default="",
        help="override rates, e.g. low=0.25,medium=0.5,high=0.75 (req/s)",
    )
    args = parser.parse_args()
    if args.request_count <= 0:
        raise SystemExit("request-count must be positive")
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    if not (args.model / "tokenizer_config.json").is_file():
        raise SystemExit(f"model tokenizer does not exist: {args.model}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    candidates, rejection_counts = load_candidates(args.dataset, tokenizer)
    selected = select_candidates(candidates, args.request_count, args.seed)
    rates = parse_rates(args.rates)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_records = []
    for item in selected:
        base_records.append(
            {
                "request_id": item["request_id"],
                "source_index": item["source_index"],
                "source_id": item["source_id"],
                "source_content_sha256": item["source_content_sha256"],
                "prompt": item["prompt"],
                "messages": item["messages"],
                "prompt_token_ids": item["prompt_token_ids"],
                "prompt_token_ids_sha256": sha256_bytes(
                    ",".join(str(token) for token in item["prompt_token_ids"]).encode(
                        "ascii"
                    )
                ),
                "prompt_tokens": item["prompt_tokens"],
                "reference_assistant_tokens": item["reference_assistant_tokens"],
                "max_tokens": item["max_tokens"],
                "input_bin": item["input_bin"],
                "output_bin": item["output_bin"],
            }
        )

    trace_hashes: dict[str, str] = {}
    normalized_arrivals_s = build_normalized_arrivals(
        len(base_records), args.seed + 1000
    )
    for trace_name, rate in rates.items():
        records = []
        for base, normalized_arrival_s in zip(base_records, normalized_arrivals_s):
            records.append(
                {
                    **base,
                    "arrival_ms": round(normalized_arrival_s * 1000.0 / rate, 6),
                    "arrival_rate_req_s": rate,
                    "arrival_shape": 0.5,
                    "normalized_arrival_s": round(normalized_arrival_s, 9),
                    "trace_name": trace_name,
                }
            )
        path = args.output_dir / f"sharegpt_qwen_trace_{trace_name}.jsonl"
        trace_hashes[trace_name] = write_jsonl(path, records)

    base_path = args.output_dir / "sharegpt_qwen_trace_base.jsonl"
    base_hash = write_jsonl(base_path, base_records)

    with (args.output_dir / "length_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["input_bin", "output_bin", "count", "prompt_tokens", "max_tokens"])
        for input_name, _, _ in INPUT_BINS:
            for output_name, _, _ in OUTPUT_BINS:
                rows = [
                    item
                    for item in base_records
                    if item["input_bin"] == input_name and item["output_bin"] == output_name
                ]
                writer.writerow(
                    [
                        input_name,
                        output_name,
                        len(rows),
                        sum(item["prompt_tokens"] for item in rows) / len(rows),
                        sum(item["max_tokens"] for item in rows) / len(rows),
                    ]
                )

    metadata = {
        "schema": "eurosys_sharegpt_open_loop_trace_v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "model": str(args.model.resolve()),
        "tokenizer_hash": tokenizer_hash(args.model),
        "seed": args.seed,
        "request_count": len(base_records),
        "selection": {
            "unit": "first human/user turn plus first assistant turn",
            "deduplication": "sha256(prompt + NUL + assistant reference)",
            "input_bins": [list(item) for item in INPUT_BINS],
            "output_bins": [list(item) for item in OUTPUT_BINS],
            "quota_per_2d_cell": args.request_count // 16,
            "max_prompt_plus_output_tokens": 8192,
            "candidate_count_after_filters": len(candidates),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "arrival_process": {
            "kind": "open_loop_gamma_interarrival",
            "shape": 0.5,
            "rates_req_s": rates,
            "first_arrival_ms": 0.0,
            "normalized_seed": args.seed + 1000,
            "normalized_interarrival_shape": 0.5,
            "normalized_arrivals_sha256": sha256_json(
                [round(value, 9) for value in normalized_arrivals_s]
            ),
            "scaled_only": True,
            "same_request_order_across_loads": True,
            "same_trace_required_for_all_mechanisms": True,
        },
        "files": {
            "base": {"path": str(base_path), "sha256": base_hash},
            **{
                name: {
                    "path": str(args.output_dir / f"sharegpt_qwen_trace_{name}.jsonl"),
                    "sha256": digest,
                }
                for name, digest in trace_hashes.items()
            },
            "length_summary": {"path": str(args.output_dir / "length_summary.csv")},
        },
        "protocol_notes": [
            "ShareGPT provides request content, not production arrival timestamps.",
            "The arrival process is synthetic and must be reported as such.",
            "Use one cold process per mechanism and do not count warmup requests.",
            "Use the same trace file, tokenizer, max_tokens, seed and scheduler limits for all arms.",
            "Calibrate rates with a pilot against the slowest arm before declaring low/medium/high utilization.",
            "The frozen prompt_token_ids are for direct-engine adapters; HTTP adapters may use messages.",
        ],
    }
    metadata_path = args.output_dir / "trace_manifest.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
