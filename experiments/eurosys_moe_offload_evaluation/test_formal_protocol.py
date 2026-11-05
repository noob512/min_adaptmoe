from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_preregisters_every_experiment() -> None:
    value = json.loads((ROOT / "protocol_template.json").read_text(encoding="utf-8"))
    assert value["schema_version"] == 1
    assert set(value["experiments"]) == {f"E{index}" for index in range(13)}
    assert value["repetition"]["mechanism_cold_processes"] >= 10
    assert value["repetition"]["system_server_processes"] >= 5
    assert value["load"]["open_loop_seconds"] >= 1200
    assert len(value["load"]["open_loop_seeds"]) >= 5
    assert value["experiments"]["E12"]["park_resume_cycles"] >= 1000


def test_packer_is_exact_and_consumes_a_prefix() -> None:
    module = load_module("formal_prepare_fixtures", ROOT / "prepare_fixtures.py")
    sources = [
        module.SourcePrompt(index, tuple(range(index + 1)), False)
        for index in range(1, 20)
    ]
    requests, used = module.pack_requests(
        sources, count=3, prompt_tokens=11, separator_id=99
    )
    assert len(requests) == 3
    assert all(len(item["prompt_token_ids"]) == 11 for item in requests)
    assert all(item["prompt_token_ids_sha256"] for item in requests)
    assert used == set(range(1, max(used) + 1))
    flattened = [index for item in requests for index in item["source_indices"]]
    assert len(flattened) == len(set(flattened))


def test_split_bucket_is_deterministic() -> None:
    module = load_module("formal_prepare_fixtures_bucket", ROOT / "prepare_fixtures.py")
    first = [module.stable_bucket(index, 20260823, 3) for index in range(100)]
    second = [module.stable_bucket(index, 20260823, 3) for index in range(100)]
    assert first == second
    assert set(first) == {0, 1, 2}


def test_shift_classifier_has_expected_scope() -> None:
    module = load_module("formal_prepare_fixtures_code", ROOT / "prepare_fixtures.py")
    assert module.CODE_RE.search("Please debug this Python function")
    assert module.CODE_RE.search("```rust\nfn main() {}\n```")
    assert not module.CODE_RE.search("Tell me a story about a quiet lake")


def test_create_run_freezes_untracked_file_contents(tmp_path: Path) -> None:
    module = load_module("formal_create_run", ROOT / "create_run.py")
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    (repository / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    script = repository / "scripts" / "run.py"
    script.parent.mkdir()
    script.write_text("print('frozen')\n", encoding="utf-8")
    script.chmod(0o755)
    (repository / "ignored.txt").write_text("not provenance\n", encoding="utf-8")
    link = repository / "run-link"
    link.symlink_to("scripts/run.py")

    files = module.list_untracked_files(repository)
    destination = tmp_path / "snapshot"
    manifest = module.snapshot_untracked_files(
        repository=repository,
        destination=destination,
        files=files,
    )

    assert Path("scripts/run.py") in files
    assert Path("run-link") in files
    assert Path("ignored.txt") not in files
    assert (destination / "scripts" / "run.py").read_text() == "print('frozen')\n"
    assert stat_mode(destination / "scripts" / "run.py") == 0o755
    assert os.readlink(destination / "run-link") == "scripts/run.py"
    records = {record["path"]: record for record in manifest["files"]}
    assert records["scripts/run.py"]["sha256"] == hashlib.sha256(
        b"print('frozen')\n"
    ).hexdigest()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o7777
