#!/usr/bin/env python3
"""CPU-only tests for the KTransformers dense FP8 tuning workflow."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("kt_dense_fp8_tuning.py")
SPEC = importlib.util.spec_from_file_location("kt_dense_fp8_tuning", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TUNING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TUNING)


class DenseFp8TuningTests(unittest.TestCase):
    def write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_warning_classes_do_not_confuse_moe_and_dense(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            log = Path(raw) / "server.log"
            prefix = "/checkout/python/sglang/srt/layers"
            log.write_text(
                "server_args=ServerArgs(fp8_gemm_runner_backend='triton')\n"
                "Using default W8A8 Block FP8 kernel config. Performance might be "
                f"sub-optimal! Config file not found at {prefix}/quantization/configs/"
                "N=5120,K=2048,device_name=NVIDIA_GeForce_RTX_5090,"
                "dtype=fp8_w8a8,block_shape=[128, 128].json\n"
                "Using default W8A8 Block FP8 kernel config. Performance might be "
                f"sub-optimal! Config file not found at {prefix}/quantization/configs/"
                "N=2048,K=4096,device_name=NVIDIA_GeForce_RTX_5090,"
                "dtype=fp8_w8a8,block_shape=[128, 128].json\n"
                "Using default MoE kernel config. Performance might be sub-optimal!\n",
                encoding="utf-8",
            )
            result = TUNING.audit_log(log, "default")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(
                result["parsed"]["faq_ignored_fused_moe_warning_count"], 1
            )
            self.assertEqual(len(result["parsed"]["dense_fallbacks"]), 2)

    def test_formal_trace_uses_mixed_chunk_dense_m(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = root / "fixture.json"
            ids = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 11, 12, 13, 14]]
            self.write_json(
                fixture,
                {
                    "metadata": {
                        "prompt_token_ids_frozen": True,
                        "prompt_token_ids_list_sha256": TUNING.sha256_json(ids),
                    },
                    "requests": [
                        {"prompt_tokens": len(values), "prompt_token_ids": values}
                        for values in ids
                    ],
                },
            )
            phases = []
            for concurrency in (1, 4):
                phases.append(
                    {
                        "concurrency": concurrency,
                        "prompt_token_ids_list_sha256": TUNING.sha256_json(
                            ids[:concurrency]
                        ),
                        "requests": [
                            {
                                "success": True,
                                "submitted_input_kind": "input_ids",
                                "server_prompt_token_count_matches_submission": True,
                                "submitted_prompt_token_ids_sha256": TUNING.sha256_json(
                                    values
                                ),
                            }
                            for values in ids[:concurrency]
                        ],
                    }
                )
            benchmark = root / "benchmark.json"
            self.write_json(
                benchmark,
                {
                    "status": "ok",
                    "config": {
                        "require_prompt_token_ids": True,
                        "output_tokens_per_request": 64,
                    },
                    "phases": phases,
                },
            )
            log = root / "server.log"
            log.write_text(
                "server_args=ServerArgs(fp8_gemm_runner_backend='triton', "
                "enable_mixed_chunk=True, disable_cuda_graph=False, "
                "disable_cuda_graph_padding=False, cuda_graph_bs=[1, 2, 4, 8])\n"
                "Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, "
                "token usage: 0.0, #running-req: 0,\n"
                "Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 0, "
                "token usage: 0.0, #running-req: 0,\n"
                "Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 0, "
                "token usage: 0.0, #running-req: 0,\n"
                "Prefill batch, #new-seq: 3, #new-token: 11, #cached-token: 0, "
                "token usage: 0.0, #running-req: 1,\n",
                encoding="utf-8",
            )
            plan = TUNING.create_plan(
                fixture, [1, 4], [f"{log}::{benchmark}"]
            )
            self.assertTrue(plan["qualification_ready"])
            self.assertEqual(plan["recommended_ms"], [1, 3, 4, 12])
            c4 = plan["observed_by_concurrency"]["4"][0]
            self.assertEqual(c4["prefill_dense_m"], [3, 12])

    def test_block_k_clamp_candidate_is_rejected(self) -> None:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
        with self.assertRaisesRegex(ValueError, "BLOCK_SIZE_K must be 128"):
            TUNING.validate_config(config)

    def test_merge_requires_both_shapes_and_preserves_exact_m_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path = root / "plan.json"
            self.write_json(
                plan_path,
                {
                    "schema": TUNING.SCHEMA,
                    "kind": "workload-m-plan",
                    "qualification_ready": True,
                    "recommended_ms": [1, 7],
                },
            )
            fragments = root / "fragments"
            fragments.mkdir()
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 3,
            }
            for n, k in TUNING.EXPECTED_SHAPES:
                for m in (1, 7):
                    self.write_json(
                        fragments / f"N{n}_K{k}_M{m}.fragment.json",
                        {
                            "schema": TUNING.SCHEMA,
                            "kind": "tuning-fragment",
                            "shape": {"M": m, "N": n, "K": k},
                            "best_config": config,
                            "environment": {
                                "device_name": TUNING.EXPECTED_DEVICE,
                                "sglang_commit": TUNING.EXPECTED_SGLANG_COMMIT,
                            },
                        },
                    )
            bundle = root / "bundle"
            manifest = TUNING.merge_fragments(plan_path, fragments, bundle)
            self.assertEqual(len(manifest["files"]), 2)
            for item in manifest["files"]:
                runtime = json.loads(
                    (bundle / "configs" / item["filename"]).read_text()
                )
                self.assertEqual(set(runtime), {"1", "7"})


if __name__ == "__main__":
    unittest.main()
