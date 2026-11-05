from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module():
    path = ROOT / "verify_formal_e2.py"
    spec = importlib.util.spec_from_file_location("verify_formal_e2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_nvml_peak_mib_uses_memory_column(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "nvml.csv"
    path.write_text(
        "2026/08/23 10:00:00.000, 50, 80, 20, 1234, 100, 200, 2400, 400, P0\n"
        "2026/08/23 10:00:00.200, 51, 90, 30, 2345, 100, 210, 2500, 500, P0\n",
        encoding="utf-8",
    )

    assert module.parse_nvml_peak_mib(path) == 2345
