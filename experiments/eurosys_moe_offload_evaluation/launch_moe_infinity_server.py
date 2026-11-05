#!/usr/bin/env python3
"""Launch MoE-Infinity's OpenAI server with an explicit offline store path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(
        token_ids, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_hf_reference(
    *, model: Any, fixture_path: Path, output_path: Path
) -> None:
    import torch

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite reference: {output_path}")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    records = fixture.get("requests")
    metadata = fixture.get("metadata")
    if not isinstance(records, list) or not records or not isinstance(metadata, dict):
        raise ValueError("reference fixture is missing requests or metadata")
    prompt_token_ids = records[0].get("prompt_token_ids")
    output_tokens = metadata.get("output_tokens")
    if (
        not isinstance(prompt_token_ids, list)
        or not prompt_token_ids
        or any(not isinstance(token_id, int) for token_id in prompt_token_ids)
        or not isinstance(output_tokens, int)
        or output_tokens <= 0
    ):
        raise ValueError("reference fixture has invalid token counts")

    input_ids = torch.tensor(
        [prompt_token_ids], dtype=torch.long, device="cuda:0"
    )
    attention_mask = torch.ones_like(input_ids)
    config = model.model.config
    eos_token_id = getattr(config, "eos_token_id", None)
    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=output_tokens,
            min_new_tokens=output_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_token_ids = [
        int(value)
        for value in generated[0, len(prompt_token_ids) :].tolist()
    ]
    if len(output_token_ids) != output_tokens:
        raise RuntimeError(
            f"HF reference generated {len(output_token_ids)}/{output_tokens} tokens"
        )

    _atomic_json(
        output_path,
        {
            "schema_version": 1,
            "status": "ok",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "path": "MoE-Infinity HuggingFace generate compatibility path",
            "model_implementation": model.model_implementation,
            "model_class_module": model.model_class_module,
            "fixture": str(fixture_path.resolve()),
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "prompt_tokens": len(prompt_token_ids),
            "output_tokens": output_tokens,
            "output_token_ids": output_token_ids,
            "output_token_ids_sha256": _token_hash(output_token_ids),
            "elapsed_seconds": elapsed,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--offload-path", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30022)
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)
    parser.add_argument("--kv-cache-ratio", type=float, default=0.01)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-tokens-per-step", type=int, default=2048)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--reference-fixture", type=Path)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()

    if not args.offload_path.is_dir():
        parser.error(f"offload path does not exist: {args.offload_path}")
    if not 0.0 < args.device_memory_ratio < 1.0:
        parser.error("--device-memory-ratio must be in (0, 1)")
    if not 0.0 <= args.kv_cache_ratio < 1.0:
        parser.error("--kv-cache-ratio must be in [0, 1)")
    if (
        args.max_batch_size <= 0
        or args.max_tokens_per_step <= 0
        or args.num_threads <= 0
    ):
        parser.error("batch size, tokens per step, and thread count must be positive")
    if (args.reference_fixture is None) != (args.reference_output is None):
        parser.error("reference fixture and output must be provided together")
    if args.reference_fixture is not None and not args.reference_fixture.is_file():
        parser.error(f"reference fixture does not exist: {args.reference_fixture}")

    from moe_infinity import MoE

    model = MoE(
        args.model,
        {
            "offload_path": str(args.offload_path.resolve()),
            "device_memory_ratio": args.device_memory_ratio,
            "num_threads": args.num_threads,
            "use_native_engine": False,
            "prefetch": False,
            "speculative_prefetch": False,
            "speculative_prefetch_overlap": False,
        },
    )
    print(
        "MODEL_IMPLEMENTATION: "
        f"{model.model_implementation} ({model.model_class_module})",
        flush=True,
    )
    if args.reference_fixture is not None and args.reference_output is not None:
        _run_hf_reference(
            model=model,
            fixture_path=args.reference_fixture,
            output_path=args.reference_output,
        )
    model.serve(
        host=args.host,
        port=args.port,
        device_memory_ratio=args.device_memory_ratio,
        kv_cache_ratio=args.kv_cache_ratio,
        max_batch_size=args.max_batch_size,
        max_tokens_per_step=args.max_tokens_per_step,
        enable_prefix_caching=False,
        offload_dir=str(args.offload_path.resolve()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
