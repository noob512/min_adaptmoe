#!/usr/bin/env python3
"""Record whether MoE-Infinity accepts the exact Qwen3 FP8 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--moe-infinity-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    root = args.moe_infinity_root.resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    if not (root / "moe_infinity" / "utils" / "quantization.py").is_file():
        raise FileNotFoundError(root / "moe_infinity/utils/quantization.py")

    sys.path.insert(0, str(root))
    import torch
    import transformers
    from transformers import AutoConfig

    from moe_infinity.utils.quantization import (
        SUPPORTED_QUANT_METHODS,
        detect_quantization,
        validate_quantization_support,
    )

    config = AutoConfig.from_pretrained(
        model, local_files_only=True, trust_remote_code=True
    )
    detected = detect_quantization(config, str(model))
    validation: dict[str, Any]
    try:
        if detected is not None:
            validate_quantization_support(detected, str(model))
        validation = {"accepted": True, "error_type": None, "error": None}
    except Exception as exc:  # The exact fail-closed result is the probe output.
        validation = {
            "accepted": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    payload = {
        "schema": "moe_infinity_qwen3_fp8_compatibility_v1",
        "status": "supported" if validation["accepted"] else "unsupported",
        "model": {
            "path": str(model),
            "config_sha256": sha256(model / "config.json"),
            "model_type": getattr(config, "model_type", None),
            "dtype": str(getattr(config, "dtype", None)),
            "quantization_config": getattr(config, "quantization_config", None),
        },
        "detected_quantization": (
            {
                "method": detected.method,
                "supported": detected.supported,
                "bits": detected.bits,
                "group_size": detected.group_size,
                "source": detected.source,
            }
            if detected is not None
            else None
        ),
        "moe_infinity": {
            "root": str(root),
            "commit": git_output(root, "rev-parse", "HEAD"),
            "supported_quant_methods": sorted(SUPPORTED_QUANT_METHODS),
            "worktree_diff_files": git_output(root, "diff", "--name-only").splitlines(),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "validation": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "validation": validation}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
