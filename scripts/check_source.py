#!/usr/bin/env python3
"""Check source syntax, project-owned paths, frozen inputs and expert plans."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments/adaptmoe_token_threshold_sharegpt_compare/inputs/sharegpt_deepseek_p1024_n128_20261024.json"
FIXTURE_SHA = "a4cc14060c94efb8c86f0524f2c49fbf85c0ff284eb12636e1e5c133d85bb9f2"
P512_FIXTURE = ROOT / "experiments/adaptmoe_token_threshold_sharegpt_compare/inputs/sharegpt_deepseek_p512_n128_20261023.json"
P512_FIXTURE_SHA = "febf1babf6981abb9f1d65cfd600e33e7a8833ad799b87e09e66cc913b8edef7"
SKIP = {".git", ".build", ".deps", ".venv", "__pycache__", ".pytest_cache", "artifacts"}
PORTABLE_ROOTS = (
    "scripts/",
    "tests/",
    "docs/",
    "experiments/adaptmoe_token_threshold_sharegpt_compare/",
)
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".h", ".hh", ".hpp", ".in", ".json", ".md",
    ".py", ".sh", ".toml", ".txt", ".yaml", ".yml", ".cmake", ".mk",
}


def is_portable_entry(relative: str, suffix: str) -> bool:
    """Apply local-path checks only to project-owned entry points and docs."""
    if "/inputs/" in f"/{relative}":
        # Frozen fixtures retain provenance paths in metadata; runners resolve
        # their location from the repository and accept model paths via env.
        return False
    return relative in {"Makefile", "README.md"} or relative.startswith(PORTABLE_ROOTS)


def is_allowed_binary_asset(relative: str) -> bool:
    """Vendor test media and NVIDIA shader tables are source-tree assets."""
    path = Path(relative)
    return (
        path.suffix.lower() in {".ico", ".mp4"}
        or "/assets/" in f"/{relative}"
        or "/shaders/" in f"/{relative}"
    )


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
        raw = path.read_bytes()
        if private_key.search(raw.decode("utf-8", errors="ignore")):
            problems.append(f"Private key marker: {relative}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            if not is_allowed_binary_asset(relative):
                problems.append(f"Unexpected binary: {relative}")
            continue
        if is_portable_entry(relative, path.suffix) and local_home.search(text):
            problems.append(f"Hardcoded home directory: {relative}")
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
    p512_value = json.loads(P512_FIXTURE.read_text())
    assert hashlib.sha256(P512_FIXTURE.read_bytes()).hexdigest() == P512_FIXTURE_SHA
    assert len(p512_value["requests"]) == 128
    assert all(
        len(record["prompt_token_ids"]) == record["prompt_tokens"] == 512
        for record in p512_value["requests"]
    )
    for manifest_name, fixture_sha in (
        ("sharegpt_deepseek_p1024_n128_20261024.manifest.json", FIXTURE_SHA),
        ("sharegpt_deepseek_p512_n128_20261023.manifest.json", P512_FIXTURE_SHA),
    ):
        manifest = json.loads((FIXTURE.parent / manifest_name).read_text())
        assert manifest["fixture_sha256"] == fixture_sha
        assert manifest["requests"] == 128
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
                      "p512_fixture_sha256": P512_FIXTURE_SHA,
                      "local_path_matches": 0, "nested_expert_plans": True}, indent=2))


if __name__ == "__main__":
    main()
