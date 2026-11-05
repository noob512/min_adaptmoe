#!/usr/bin/env python3
"""Fetch pinned optional baselines and apply the recorded measurement patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(directory: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(directory), *args], check=check,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> None:
    versions = json.loads((ROOT / "baselines/versions.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", metavar="BASELINE",
                        help="KTransformers or MoE-Infinity; default: both")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the selected pinned versions without fetching")
    args = parser.parse_args()
    unknown = set(args.names) - versions.keys()
    if unknown:
        parser.error("Unknown baseline(s): " + ", ".join(sorted(unknown)))
    selected = args.names or list(versions)
    if args.dry_run:
        print(json.dumps({name: versions[name] for name in selected}, indent=2))
        return
    for name in selected:
        spec = versions[name]
        destination = ROOT / name
        if destination.exists() and not (destination / ".git").exists():
            if any(destination.iterdir()):
                raise SystemExit(f"Refusing to replace non-Git directory: {destination}")
        if not (destination / ".git").exists():
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                            spec["url"], str(destination)], check=True)
            git(destination, "checkout", "--detach", spec["commit"])
        head = git(destination, "rev-parse", "HEAD").stdout.strip()
        if head != spec["commit"]:
            raise SystemExit(f"Unexpected {name} HEAD: {head}; expected {spec['commit']}")
        git(destination, "submodule", "update", "--init", "--recursive")
        for relative, expected in spec.get("nested", {}).items():
            actual = git(destination / relative, "rev-parse", "HEAD").stdout.strip()
            if actual != expected:
                raise SystemExit(f"Unexpected {name}/{relative} HEAD: {actual}")
        for relative, patch in spec.get("patches", {}).items():
            target = destination / relative
            patch_path = ROOT / "baselines" / patch
            if git(target, "apply", "--reverse", "--check", str(patch_path), check=False).returncode:
                git(target, "apply", "--check", str(patch_path))
                git(target, "apply", str(patch_path))
        print(f"{name}: {head}")


if __name__ == "__main__":
    main()
