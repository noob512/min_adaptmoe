#!/usr/bin/env python3
"""Check source syntax, portability, frozen inputs and expert plans without a GPU."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments/adaptmoe_token_threshold_sharegpt_compare/inputs/sharegpt_deepseek_p1024_n128_20261024.json"
FIXTURE_SHA = "fed1428ffe8d2fe7a4608b626041f31b81f22071876e330b495e3db3b8376a43"
SKIP = {".git", ".build", ".deps", ".venv", "__pycache__", ".pytest_cache", "artifacts"}


def main() -> None:
    paths = [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP)]
    problems: list[str] = []
    local_home = re.compile(r"/(?:home|Users)/[^/\s\"']+")
    private_key = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    python_files = 0
    for path in paths:
        relative = str(path.relative_to(ROOT))
        if path.is_symlink():
            problems.append(f"Unexpected symlink: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"Unexpected binary: {relative}")
            continue
        if local_home.search(text):
            problems.append(f"Hardcoded home directory: {relative}")
        if private_key.search(text):
            problems.append(f"Private key marker: {relative}")
        if path.suffix == ".py":
            ast.parse(text, filename=relative)
            python_files += 1
        if path.suffix == ".sh" and (relative.startswith("scripts/") or relative.startswith("experiments/")):
            subprocess.run(["bash", "-n", str(path)], check=True)
    value = json.loads(FIXTURE.read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    assert len(value["requests"]) == 128
    hashes = []
    for record in value["requests"]:
        tokens = record["prompt_token_ids"]
        digest = hashlib.sha256(",".join(map(str, tokens)).encode("ascii")).hexdigest()
        assert len(tokens) == record["prompt_tokens"] == 1024
        assert digest == record["prompt_token_ids_sha256"]
        hashes.append(digest)
    assert len(set(hashes)) == 128
    base = ROOT / "experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets"
    cold8 = json.loads((base / "plans/cold_plan_8_per_layer.json").read_text())
    cold11 = json.loads((base / "plans_adjusted_11_17_23_29/cold_plan_2g.json").read_text())
    for layer in range(1, 27):
        left, right = set(cold8["selected_by_layer"][str(layer)]), set(cold11["selected_by_layer"][str(layer)])
        assert len(left) == 8 and len(right) == 11 and left <= right
    assert cold8["selected_main_bytes"] == 3598712832
    assert cold11["selected_main_bytes"] == 4948230144
    if problems:
        raise SystemExit("\n".join(problems))
    print(json.dumps({"status": "ok", "source_files": len(paths), "python_syntax_checked": python_files,
                      "requests": 128, "prompt_tokens": 1024, "fixture_sha256": FIXTURE_SHA,
                      "local_path_matches": 0, "nested_expert_plans": True}, indent=2))


if __name__ == "__main__":
    main()
