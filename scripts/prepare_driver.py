#!/usr/bin/env python3
"""Fetch a pinned NVIDIA source tree and overlay the reviewed UVM sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "595.71.05"
COMMIT = "095de56d46690f857d7a5aa2512eadee71ed013f"
ARCHIVE_SHA256 = "836c03b5445e04975a54c2afc76cca7403e9abb4c13f71d02037bb0672692b21"
URL = f"https://codeload.github.com/NVIDIA/open-gpu-kernel-modules/tar.gz/{COMMIT}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".build" / f"nvidia-{VERSION}")
    parser.add_argument("--source-archive", type=Path, help="Use an already downloaded archive with the same checksum")
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists():
        parser.error(f"Output exists; choose a fresh directory: {output}")
    overlay = ROOT / "kernel-module/nvidia-module/kernel-open/nvidia-uvm"
    if not (overlay / "uvm_va_block.c").is_file():
        parser.error("Reviewed UVM overlay is missing")
    with tempfile.TemporaryDirectory(prefix="adaptmoe-driver-") as temporary:
        staging = Path(temporary)
        archive_path = args.source_archive
        if archive_path is None:
            archive_path = staging / "upstream.tar.gz"
            with urllib.request.urlopen(URL, timeout=60) as response, archive_path.open("wb") as target:
                shutil.copyfileobj(response, target)
        if sha256(archive_path) != ARCHIVE_SHA256:
            raise SystemExit("NVIDIA archive checksum mismatch")
        with tarfile.open(archive_path) as archive:
            archive.extractall(staging / "extracted", filter="data")
        upstream = staging / "extracted" / f"open-gpu-kernel-modules-{COMMIT}"
        if not (upstream / "Makefile").is_file():
            raise SystemExit("Unexpected NVIDIA archive layout")
        destination = upstream / "kernel-open/nvidia-uvm"
        shutil.rmtree(destination)
        shutil.copytree(overlay, destination)
        receipt = {
            "upstream_version": VERSION,
            "upstream_commit": COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "upstream_url": URL,
            "uvm_source_sha256": {
                str(path.relative_to(overlay)): sha256(path)
                for path in sorted(overlay.rglob("*")) if path.is_file()
            },
        }
        (upstream / "review_build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(upstream), str(output))
    print(output)


if __name__ == "__main__":
    main()
