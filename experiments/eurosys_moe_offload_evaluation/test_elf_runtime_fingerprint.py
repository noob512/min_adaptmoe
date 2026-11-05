from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("elf_runtime_fingerprint.py")
SPEC = importlib.util.spec_from_file_location("elf_runtime_fingerprint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rejects_non_elf(tmp_path: Path) -> None:
    path = tmp_path / "not-elf"
    path.write_bytes(b"not an ELF file")
    try:
        MODULE.fingerprint(path)
    except ValueError as exc:
        assert "ELF" in str(exc)
    else:
        raise AssertionError("non-ELF input was accepted")


def test_full_file_and_runtime_hashes_are_reported() -> None:
    result = MODULE.fingerprint(Path("/bin/true"))
    assert result["schema_version"] == 1
    assert len(result["file_sha256"]) == 64
    assert len(result["runtime_elf_sha256"]) == 64
    assert result["sections"]
    assert all(record["flags"] & MODULE.SHF_ALLOC for record in result["sections"])
