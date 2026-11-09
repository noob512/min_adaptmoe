#!/usr/bin/env python3
"""Link ignored build inputs from a compatible local NVIDIA/vLLM workspace.

Only the Python environment, shared libraries and UVM module are borrowed.
The experiment runners continue to import this checkout's Python source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--baselines", action="store_true",
                        help="also link the pinned, already built optional baselines")
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    if workspace == ROOT:
        parser.error("Choose a different workspace")
    paths = [Path("workloads/vllm/vllm/.venv"),
             Path("kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko")]
    vllm = workspace / "workloads/vllm/vllm"
    for directory, children, filenames in os.walk(vllm):
        children[:] = [n for n in children if n not in
                       {".venv", ".git", "build", ".build", ".cache", ".deps", "__pycache__"}]
        paths.extend((Path(directory) / name).relative_to(workspace)
                     for name in filenames if name.endswith(".so"))
    required = {Path("workloads/vllm/vllm/uvm_test/uvm_allocator.so"),
                Path("workloads/vllm/vllm/vllm/_C.abi3.so")}
    if not required <= set(paths):
        parser.error("Workspace lacks the compiled allocator or vLLM extension")
    if args.baselines:
        versions = json.loads((ROOT / "baselines/versions.json").read_text())
        for name, version in versions.items():
            source = workspace / name
            actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if actual != version["commit"]:
                parser.error(f"{name} revision differs: {actual}")
            for nested, expected in version.get("nested", {}).items():
                actual = subprocess.check_output(["git", "-C", str(source / nested), "rev-parse", "HEAD"], text=True).strip()
                if actual != expected:
                    parser.error(f"{name}/{nested} revision differs: {actual}")
            paths.append(Path(name))
    # Validate every destination before creating any links; never replace files.
    for relative in paths:
        source, destination = workspace / relative, ROOT / relative
        if not source.exists():
            parser.error(f"Missing build input: {source}")
        if (destination.exists() or destination.is_symlink()) and (
                not destination.is_symlink() or destination.resolve() != source.resolve()):
            parser.error(f"Refusing to replace {destination}")
    records = []
    for relative in paths:
        source, destination = workspace / relative, ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_symlink():
            destination.symlink_to(source, target_is_directory=source.is_dir())
        records.append({"path": str(relative), "external_path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None})
    output = ROOT / ".build/borrowed-runtime.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"workspace": str(workspace), "links": records}, indent=2) + "\n")
    print(f"Linked {len(records)} build inputs; receipt: {output}")


if __name__ == "__main__":
    main()
