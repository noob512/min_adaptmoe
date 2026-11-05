#!/usr/bin/env python3
"""Create a non-overwritable EuroSys run root and freeze provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TEMPLATE = SCRIPT_DIR / "protocol_template.json"
PLAN = REPO_ROOT / "docs/eurosys_moe_offload_evaluation_plan.md"
TECHNICAL_DESIGN = REPO_ROOT / "docs/moe_phase_aware_offload_technical_design.md"
FRESHNESS_EPOCH = datetime(2026, 8, 23, tzinfo=timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*command: str, cwd: Path | None = None, check: bool = True) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def repository_record(path: Path) -> dict[str, Any]:
    commit = run("git", "rev-parse", "HEAD", cwd=path).strip()
    status = run("git", "status", "--porcelain=v1", cwd=path)
    patch = run("git", "diff", "--binary", "HEAD", cwd=path)
    untracked = sorted(
        line[3:] for line in status.splitlines() if line.startswith("?? ")
    )
    return {
        "path": str(path.resolve()),
        "commit": commit,
        "dirty": bool(status),
        "status": status,
        "tracked_patch": patch,
        "untracked_paths": untracked,
    }


def list_untracked_files(
    repository: Path, *, exclude_root: Path | None = None
) -> list[Path]:
    """Enumerate every non-ignored untracked file without directory folding."""
    completed = subprocess.run(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    excluded = exclude_root.resolve() if exclude_root is not None else None
    result: list[Path] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe untracked path from git: {relative}")
        source = repository / relative
        if excluded is not None and source.resolve().is_relative_to(excluded):
            continue
        mode = source.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            raise RuntimeError(
                f"unsupported untracked filesystem entry: {source}"
            )
        result.append(relative)
    return sorted(result, key=lambda value: os.fsencode(value.as_posix()))


def _symlink_sha256(path: Path) -> str:
    return hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()


def snapshot_untracked_files(
    *, repository: Path, destination: Path, files: list[Path]
) -> dict[str, Any]:
    """Copy exact untracked content into provenance and return its manifest."""
    destination.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for relative in files:
        source = repository / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_stat = source.lstat()
        mode = stat.S_IMODE(source_stat.st_mode)
        if stat.S_ISLNK(source_stat.st_mode):
            link_target = os.readlink(source)
            target.symlink_to(link_target)
            digest = _symlink_sha256(source)
            kind = "symlink"
            record: dict[str, Any] = {
                "path": relative.as_posix(),
                "kind": kind,
                "mode": f"{mode:04o}",
                "size_bytes": source_stat.st_size,
                "sha256": digest,
                "link_target": link_target,
            }
        elif stat.S_ISREG(source_stat.st_mode):
            shutil.copy2(source, target, follow_symlinks=False)
            digest = sha256_file(source)
            if sha256_file(target) != digest:
                raise RuntimeError(f"untracked snapshot verification failed: {source}")
            kind = "file"
            record = {
                "path": relative.as_posix(),
                "kind": kind,
                "mode": f"{mode:04o}",
                "size_bytes": source_stat.st_size,
                "sha256": digest,
            }
        else:
            raise RuntimeError(f"unsupported untracked entry: {source}")
        total_bytes += source_stat.st_size
        records.append(record)
    return {
        "source_repository": str(repository.resolve()),
        "snapshot_root": str(destination.resolve()),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "files": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--fixture-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{5,127}", args.protocol_id):
        raise SystemExit("invalid protocol ID")
    if args.run_root.exists():
        raise SystemExit(f"run root already exists: {args.run_root}")
    now = datetime.now(timezone.utc)
    if now < FRESHNESS_EPOCH:
        raise SystemExit("system clock predates the formal-data freshness epoch")
    template = json.loads(args.template.read_text(encoding="utf-8"))
    if template.get("schema_version") != 1:
        raise SystemExit("unsupported protocol template")
    if args.fixture_manifest is not None:
        fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
        if fixture.get("status") != "ok" or fixture.get("protocol_id") != args.protocol_id:
            raise SystemExit("fixture manifest does not match the protocol")
    else:
        fixture = None

    provenance = args.run_root / "provenance"
    provenance.mkdir(parents=True)
    (args.run_root / "cases").mkdir()
    protocol = {
        **template,
        "protocol_id": args.protocol_id,
        "frozen_at": now.isoformat(),
        "freshness_epoch": FRESHNESS_EPOCH.isoformat(),
        "template_sha256": sha256_file(args.template),
        "evaluation_plan_sha256": sha256_file(PLAN),
        "technical_design_sha256": sha256_file(TECHNICAL_DESIGN),
        "fixture_manifest": (
            {
                "path": str(args.fixture_manifest.resolve()),
                "sha256": sha256_file(args.fixture_manifest),
            }
            if args.fixture_manifest is not None
            else None
        ),
    }
    atomic_json(args.run_root / "protocol.json", protocol)

    repository_paths = {
        "main": REPO_ROOT,
        "ktransformers": REPO_ROOT / "KTransformers",
        "sglang": REPO_ROOT / "KTransformers/third_party/sglang",
        "moe_infinity": REPO_ROOT / "MoE-Infinity",
    }
    repositories = {
        name: repository_record(path)
        for name, path in repository_paths.items()
    }
    compact: dict[str, Any] = {}
    patches = provenance / "dirty_patches"
    patches.mkdir()
    untracked_snapshots = provenance / "untracked_files"
    untracked_manifests: dict[str, Any] = {}
    for name, value in repositories.items():
        status_path = patches / f"{name}.status.txt"
        patch_path = patches / f"{name}.tracked.patch"
        status_path.write_text(value.pop("status"), encoding="utf-8")
        patch_path.write_text(value.pop("tracked_patch"), encoding="utf-8")
        value["status_sha256"] = sha256_file(status_path)
        value["tracked_patch_sha256"] = sha256_file(patch_path)
        untracked_files = list_untracked_files(
            repository_paths[name], exclude_root=args.run_root
        )
        untracked_manifest = snapshot_untracked_files(
            repository=repository_paths[name],
            destination=untracked_snapshots / name,
            files=untracked_files,
        )
        value["untracked_file_count"] = untracked_manifest["file_count"]
        value["untracked_total_bytes"] = untracked_manifest["total_bytes"]
        untracked_manifests[name] = untracked_manifest
        compact[name] = value
    atomic_json(
        provenance / "untracked_files_manifest.json", untracked_manifests
    )
    atomic_json(provenance / "git_commits.json", compact)

    commands = {
        "uname": ("uname", "-a"),
        "lscpu": ("lscpu",),
        "numa": ("numactl", "--hardware"),
        "nvidia_smi": ("nvidia-smi", "-q"),
        "memory": ("free", "-h"),
        "filesystem": ("df", "-hT", str(REPO_ROOT)),
    }
    environment_parts = [
        f"created_at={now.isoformat()}\n",
        f"python={platform.python_version()}\n",
    ]
    for label, command in commands.items():
        environment_parts.append(f"\n[{label}]\n")
        environment_parts.append(run(*command, check=False))
    (provenance / "environment.txt").write_text(
        "".join(environment_parts), encoding="utf-8"
    )

    model_paths = {
        "deepseek_v2_lite_bf16": Path(
            "/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--"
            "DeepSeek-V2-Lite-Chat/snapshots/"
            "85864749cd611b4353ce1decdb286193298f64c7"
        ),
        "qwen3_30b_a3b_fp8": Path(
            "/home/ubuntu/.cache/huggingface/hub/models--Qwen--"
            "Qwen3-30B-A3B-FP8/snapshots/"
            "d206ba732169f29bb77fbf80fc2c4b81d4d30782"
        ),
    }
    models: dict[str, Any] = {}
    for name, model in model_paths.items():
        files = {}
        for filename in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
            path = model / filename
            if path.is_file():
                files[filename] = {"path": str(path), "sha256": sha256_file(path)}
        models[name] = {"snapshot": str(model), "files": files}
    atomic_json(provenance / "model_manifest.json", models)
    if fixture is not None:
        atomic_json(provenance / "fixture_manifest.json", fixture)

    marker = {
        "schema_version": 1,
        "protocol_id": args.protocol_id,
        "status": "preregistered",
        "created_at": now.isoformat(),
        "verification_status": "not_run",
    }
    atomic_json(args.run_root / "run_status.json", marker)
    print(args.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
