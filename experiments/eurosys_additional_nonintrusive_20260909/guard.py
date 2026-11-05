#!/usr/bin/env python3
"""Freeze and verify the pre-existing repository state.

The directory containing this file is the only allowed repository prefix.
Pre-existing tracked changes and untracked files are user data and are hashed
as immutable inputs rather than treated as a clean-worktree failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALLOWED = HERE.relative_to(ROOT).as_posix() + "/"


def run(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode("utf-8", "replace")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_path(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return {"kind": "file", "size": path.stat().st_size, "sha256": digest.hexdigest()}
    if path.is_dir():
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        ).stdout
        return {"kind": "directory", "git_head": head, "git_status_sha256": sha_bytes(status.encode())}
    return {"kind": "missing"}


def status_lines() -> list[str]:
    value = run("git", "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(value, str)
    lines = []
    for line in value.splitlines():
        path = line[3:]
        if path.startswith(ALLOWED):
            continue
        lines.append(line)
    return lines


def status_paths(lines: list[str]) -> list[str]:
    paths: list[str] = []
    for line in lines:
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value)
    return sorted(set(paths))


def capture() -> dict[str, Any]:
    lines = status_lines()
    worktree_diff = run("git", "diff", "--binary", "HEAD", "--", ".", binary=True)
    cached_diff = run("git", "diff", "--binary", "--cached", "HEAD", "--", ".", binary=True)
    assert isinstance(worktree_diff, bytes) and isinstance(cached_diff, bytes)
    dirty_hashes = {}
    for relative in status_paths(lines):
        dirty_hashes[relative] = sha_path(ROOT / relative)
    return {
        "head": str(run("git", "rev-parse", "HEAD")).strip(),
        "allowed_prefix": ALLOWED,
        "status_lines": lines,
        "worktree_diff_sha256": sha_bytes(worktree_diff),
        "cached_diff_sha256": sha_bytes(cached_diff),
        "dirty_path_hashes": dirty_hashes,
    }


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "check"))
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="unspecified")
    args = parser.parse_args()

    current = capture()
    if args.action == "init":
        value = {
            "schema": "nonintrusive_git_guard_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state": current,
        }
        write_new(args.baseline, value)
        print(args.baseline)
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["state"]
    equal = current == baseline
    value = {
        "schema": "nonintrusive_git_guard_check_v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "ok": equal,
        "baseline": baseline,
        "current": current,
    }
    if args.output:
        write_new(args.output, value)
    print(json.dumps({"label": args.label, "ok": equal}, sort_keys=True))
    return 0 if equal else 2


if __name__ == "__main__":
    raise SystemExit(main())

