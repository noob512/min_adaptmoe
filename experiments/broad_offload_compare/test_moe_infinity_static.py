#!/usr/bin/env python3
"""CPU-only tests for MoE-Infinity qualification assets and evidence gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MOEI_ROOT = REPO_ROOT / "MoE-Infinity"
PYTHON = REPO_ROOT / "workloads/vllm/vllm/.venv/bin/python"
MODEL = Path(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/"
    "85864749cd611b4353ce1decdb286193298f64c7"
)
EVALUATION_FIXTURE = (
    SCRIPT_DIR / "fixtures/moei_deepseek_v2_lite_evaluation_prompt512_v1.json"
)
CALIBRATION_FIXTURE = (
    SCRIPT_DIR / "fixtures/moei_deepseek_v2_lite_calibration_prompt512_v1.json"
)
EXPECTED_FILE_HASHES = {
    EVALUATION_FIXTURE: (
        "5315aa367325a0adea8ac30fc7d16659bd0bdcbcfe608f8f5c3d59ea7d415bd5"
    ),
    CALIBRATION_FIXTURE: (
        "14ce7cb4960109fc3744ecd10fbcbb7a8c8fca41a82ee89b7de5a118fd4eaffe"
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeInstrumentation:
    def __init__(self, counters: dict[str, int]) -> None:
        self.counters = counters

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)


class StaticQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qualifier = load_module(
            "qualify_moe_infinity_static",
            SCRIPT_DIR / "qualify_moe_infinity.py",
        )

    def test_frozen_fixtures_are_valid_and_distinct(self) -> None:
        token_hashes = set()
        for path, expected_file_hash in EXPECTED_FILE_HASHES.items():
            self.assertEqual(sha256_file(path), expected_file_hash)
            value = json.loads(path.read_text(encoding="utf-8"))
            ids = value["prompt_token_ids"]
            self.assertEqual(len(ids), 512)
            self.assertTrue(all(type(item) is int and item >= 0 for item in ids))
            encoded = ",".join(str(item) for item in ids).encode("ascii")
            token_hash = hashlib.sha256(encoded).hexdigest()
            self.assertEqual(value["prompt_token_ids_sha256"], token_hash)
            self.assertEqual(
                value["model_snapshot_name"],
                "85864749cd611b4353ce1decdb286193298f64c7",
            )
            token_hashes.add(token_hash)
        self.assertEqual(len(token_hashes), 2)

    def test_fixture_freezer_is_byte_deterministic(self) -> None:
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for role, expected in (
            ("evaluation", EVALUATION_FIXTURE),
            ("calibration", CALIBRATION_FIXTURE),
        ):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "fixture.json"
                subprocess.run(
                    [
                        str(PYTHON),
                        str(SCRIPT_DIR / "freeze_moe_infinity_prompt_ids.py"),
                        "--model",
                        str(MODEL),
                        "--role",
                        role,
                        "--prompt-length",
                        "512",
                        "--output",
                        str(output),
                    ],
                    check=True,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    timeout=60,
                )
                self.assertEqual(output.read_bytes(), expected.read_bytes())

    def _record(self, index: int, warmup: bool, output_hash: str = "same") -> dict:
        zero = {
            "prediction_calls": 0,
            "correction_calls": 0,
            "enqueue_attempts": 0,
            "enqueue_errors": 0,
            "requested_effective_layer_differences": 0,
        }
        return {
            "trial_index": index,
            "warmup": warmup,
            "prompt_tokens": 512,
            "output_tokens": 32,
            "generated_token_ids_sha256": output_hash,
            "prefetch_counter_delta": zero,
        }

    def test_demand_lifecycle_gate_accepts_exact_zero_path(self) -> None:
        zero = {
            "prediction_calls": 0,
            "correction_calls": 0,
            "enqueue_attempts": 0,
            "enqueue_errors": 0,
            "requested_effective_layer_differences": 0,
        }
        checks, failures = self.qualifier.qualification_checks(
            "demand-only",
            self.qualifier.MODES["demand-only"],
            [self._record(index, True) for index in range(3)],
            [self._record(index, False) for index in range(5)],
            FakeInstrumentation(zero),
            False,
            512,
            32,
            3,
            5,
        )
        self.assertEqual(failures, [])
        self.assertEqual(checks["status"], "ok")

    def test_demand_lifecycle_gate_rejects_warmup_prefetch(self) -> None:
        counters = {
            "prediction_calls": 0,
            "correction_calls": 0,
            "enqueue_attempts": 1,
            "enqueue_errors": 0,
            "requested_effective_layer_differences": 0,
        }
        checks, failures = self.qualifier.qualification_checks(
            "demand-only",
            self.qualifier.MODES["demand-only"],
            [self._record(index, True) for index in range(3)],
            [self._record(index, False) for index in range(5)],
            FakeInstrumentation(counters),
            False,
            512,
            32,
            3,
            5,
        )
        self.assertEqual(checks["status"], "error")
        self.assertTrue(any("warmup or measurement" in item for item in failures))

    def test_demand_lifecycle_gate_rejects_output_drift(self) -> None:
        zero = {
            "prediction_calls": 0,
            "correction_calls": 0,
            "enqueue_attempts": 0,
            "enqueue_errors": 0,
            "requested_effective_layer_differences": 0,
        }
        warmups = [self._record(index, True) for index in range(3)]
        warmups[0]["generated_token_ids_sha256"] = "warmup-drift"
        checks, failures = self.qualifier.qualification_checks(
            "demand-only",
            self.qualifier.MODES["demand-only"],
            warmups,
            [self._record(index, False) for index in range(5)],
            FakeInstrumentation(zero),
            False,
            512,
            32,
            3,
            5,
        )
        self.assertEqual(checks["status"], "error")
        self.assertTrue(any("warmup and measured" in item for item in failures))

    def test_patch_scope_and_clean_import_blocker(self) -> None:
        changed = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=MOEI_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(
            changed,
            [
                "core/parallel/expert_module.cpp",
                "moe_infinity/models/__init__.py",
                "moe_infinity/runtime/model_offload.py",
                "setup.py",
            ],
        )
        base_models_init = subprocess.run(
            ["git", "show", "HEAD:moe_infinity/models/__init__.py"],
            cwd=MOEI_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn(
            "from .qwen3_5_moe import SyncQwen3_5MoeSparseMoeBlock",
            base_models_init,
        )
        transformers_models = next(
            (REPO_ROOT / "workloads/vllm/vllm/.venv/lib").glob(
                "python*/site-packages/transformers/models"
            )
        )
        self.assertFalse((transformers_models / "qwen3_5_moe").exists())
        patch_sha256 = hashlib.sha256(
            subprocess.run(
                ["git", "diff", "--binary"],
                cwd=MOEI_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        ).hexdigest()
        runner = (SCRIPT_DIR / "run_moe_infinity_qualification.sh").read_text()
        self.assertIn(
            f'EXPECTED_MOEI_PATCH_SHA256="{patch_sha256}"',
            runner,
        )
        self.assertIn('STORE_BUILD_RECEIPT="${STORE_BUILD_RECEIPT:-}"', runner)
        self.assertNotIn("moe_infinity_minimal_store_build_20260813", runner)
        extension = next((MOEI_ROOT / "moe_infinity").glob("_store*.so"))
        runtime_fingerprinter = (
            REPO_ROOT
            / "experiments/eurosys_moe_offload_evaluation/elf_runtime_fingerprint.py"
        )
        extension_runtime_sha256 = subprocess.run(
            [str(PYTHON), str(runtime_fingerprinter), str(extension)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertIn(
            f'EXPECTED_STORE_RUNTIME_SHA256="{extension_runtime_sha256}"',
            runner,
        )

    def test_cpp_patch_preserves_formula_and_upstream_small_path(self) -> None:
        current = (MOEI_ROOT / "core/parallel/expert_module.cpp").read_text()
        base = subprocess.run(
            ["git", "show", "HEAD:core/parallel/expert_module.cpp"],
            cwd=MOEI_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        current_formula = current[current.index("void MoEMLP::ForwardHelper") :]
        base_formula = base[base.index("void MoEMLP::ForwardHelper") :]
        self.assertEqual(current_formula, base_formula)
        current_forward = current[
            current.index("torch::Tensor MoEMLP::forward") : current.index(
                "void MoEMLP::ForwardHelper"
            )
        ]
        base_forward = base[
            base.index("torch::Tensor MoEMLP::forward") : base.index(
                "void MoEMLP::ForwardHelper"
            )
        ]
        small_path = current_forward[
            current_forward.index("if (batch_size <= kMaxTokens) {") :
            current_forward.index(
                "// Compatibility fallback for inputs beyond the upstream"
            )
        ]
        fallback = current_forward[
            current_forward.index(
                "// Compatibility fallback for inputs beyond the upstream"
            ) :
        ]
        self.assertEqual(base_forward.count("cudaStreamSynchronize"), 2)
        self.assertEqual(small_path.count("cudaStreamSynchronize"), 2)
        self.assertIn("auto output = output_.clone();", small_path)
        self.assertNotIn("torch::empty_like", small_path)
        self.assertEqual(fallback.count("cudaStreamSynchronize"), 3)
        self.assertIn("auto output = torch::empty_like(hidden_states);", fallback)
        self.assertIn("for (int64_t offset = 0;", fallback)
        self.assertNotIn("for (int64_t offset = 0;", base_forward)


if __name__ == "__main__":
    unittest.main(verbosity=2)
