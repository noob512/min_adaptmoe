#!/usr/bin/env python3
"""CPU-only tests for the KT dense FP8 D-T-T-D controller."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("kt_dense_fp8_ablation.py")
SPEC = importlib.util.spec_from_file_location("kt_dense_fp8_ablation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class DenseFp8AblationTests(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        bundle = root / "bundle"
        records = []
        for filename in sorted(ablation.expected_filenames()):
            config = {
                "1": {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2,
                },
                "8": {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2,
                },
            }
            config_path = bundle / "configs" / filename
            write_json(config_path, config)
            records.append(
                {
                    "filename": filename,
                    "sha256": ablation.sha256_file(config_path),
                    "M_keys": [1, 8],
                }
            )
        write_json(
            bundle / "manifest.json",
            {
                "schema": ablation.TUNING_SCHEMA,
                "kind": "installable-bundle",
                "expected_kt_commit": ablation.EXPECTED_KT_COMMIT,
                "expected_sglang_commit": ablation.EXPECTED_SGLANG_COMMIT,
                "device_name": "NVIDIA GeForce RTX 5090",
                "activation_output_dtype": "bfloat16",
                "files": records,
            },
        )
        return bundle

    def make_checkout(self, root: Path, name: str = "KTransformers") -> Path:
        checkout = root / name
        ablation.config_target_dir(checkout).mkdir(parents=True)
        return checkout

    def make_fixture(self, path: Path, offset: int) -> Path:
        ids = [[offset + index, offset + index + 100] for index in range(8)]
        write_json(
            path,
            {
                "metadata": {
                    "prompt_token_ids_frozen": True,
                    "prompt_token_ids_list_sha256": ablation.sha256_json(ids),
                },
                "requests": [
                    {"prompt_token_ids": values, "prompt_tokens": len(values)}
                    for values in ids
                ],
            },
        )
        return path

    def make_inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        evaluation = self.make_fixture(root / "evaluation.json", 10)
        warmup = self.make_fixture(root / "warmup.json", 1000)
        stats = root / "frequency_stats.pt"
        stats.write_bytes(b"synthetic-frequency-stats")
        stats_manifest = root / "frequency_stats_manifest.json"
        write_json(
            stats_manifest,
            {
                "placement": {
                    "gpu_mask_uint8_sha256": ablation.EXPECTED_MASK_SHA,
                    "hot_experts": ablation.EXPECTED_HOT_EXPERTS,
                    "cold_experts": ablation.EXPECTED_COLD_EXPERTS,
                },
                "normalized_stats": {
                    "path": str(stats.resolve()),
                    "sha256": ablation.sha256_file(stats),
                },
            },
        )
        return evaluation, warmup, stats, stats_manifest

    def preregister(self, root: Path) -> tuple[Path, Path, dict[str, object]]:
        bundle = self.make_bundle(root)
        checkout = self.make_checkout(root)
        evaluation, warmup, stats, stats_manifest = self.make_inputs(root)
        run_root = root / "run"
        args = argparse.Namespace(
            run_root=run_root,
            bundle_dir=bundle,
            kt_root=checkout,
            evaluation_fixture=evaluation,
            warmup_fixture=warmup,
            frequency_stats=stats,
            stats_manifest=stats_manifest,
        )

        def fake_git_head(path: Path) -> str:
            return (
                ablation.EXPECTED_SGLANG_COMMIT
                if Path(path).name == "sglang"
                else ablation.EXPECTED_KT_COMMIT
            )

        with mock.patch.object(ablation, "git_head", side_effect=fake_git_head):
            manifest = ablation.preregister(args)
        return run_root, checkout, manifest

    def test_transition_round_trip_and_hash_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.make_bundle(root)
            checkout = self.make_checkout(root)
            validated = ablation.validate_bundle(bundle)
            self.assertEqual(ablation.inspect_config_state(bundle, checkout)["state"], "default")

            install = ablation.transition_config(
                bundle_dir=bundle,
                kt_root=checkout,
                target_state="tuned",
                journal=root / "install.json",
                verify_checkout=False,
            )
            self.assertEqual(install["before"]["state"], "default")
            self.assertEqual(install["after"]["state"], "tuned")
            self.assertEqual(
                install["bundle"]["manifest_sha256"], validated["manifest_sha256"]
            )
            for item in install["after"]["files"]:
                self.assertEqual(item["actual_sha256"], item["bundle_sha256"])

            remove = ablation.transition_config(
                bundle_dir=bundle,
                kt_root=checkout,
                target_state="default",
                journal=root / "remove.json",
                verify_checkout=False,
            )
            self.assertEqual(remove["before"]["state"], "tuned")
            self.assertEqual(remove["after"]["state"], "default")
            self.assertEqual(ablation.inspect_config_state(bundle, checkout)["state"], "default")

    def test_partial_and_different_targets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.make_bundle(root)
            validated = ablation.validate_bundle(bundle)
            for mode in ("partial", "different"):
                with self.subTest(mode=mode):
                    checkout = self.make_checkout(root, f"checkout-{mode}")
                    target_dir = ablation.config_target_dir(checkout)
                    records = validated["files"][:1] if mode == "partial" else validated["files"]
                    for record in records:
                        target = target_dir / record["filename"]
                        if mode == "partial":
                            target.write_bytes(Path(record["source"]).read_bytes())
                        else:
                            target.write_bytes(b"different-config")
                    self.assertEqual(
                        ablation.inspect_config_state(bundle, checkout)["state"], "invalid"
                    )
                    with self.assertRaisesRegex(ValueError, "partial or differs"):
                        ablation.transition_config(
                            bundle_dir=bundle,
                            kt_root=checkout,
                            target_state="tuned",
                            journal=root / f"{mode}.json",
                            verify_checkout=False,
                        )

    def test_preregister_writes_all_slots_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root, _, manifest = self.preregister(Path(temporary))
            self.assertEqual(
                [(item["slot"], item["arm"], item["config_state"]) for item in manifest["schedule"]],
                list(ablation.SCHEDULE),
            )
            status = ablation.load_json(run_root / "execution_status.json")
            self.assertEqual(len(status["slots"]), 4)
            self.assertEqual({item["status"] for item in status["slots"]}, {"preregistered"})
            self.assertTrue(all(item["returncode"] is None for item in status["slots"]))
            self.assertTrue((run_root / "bundle_snapshot/manifest.json").is_file())
            self.assertTrue((run_root / "abba_order.tsv").is_file())

    def config_state(
        self, bundle: dict[str, object], checkout: Path, state: str
    ) -> dict[str, object]:
        files = []
        for record in bundle["files"]:
            exists = state == "tuned"
            files.append(
                {
                    "filename": record["filename"],
                    "target": str(ablation.config_target_dir(checkout) / record["filename"]),
                    "exists": exists,
                    "actual_sha256": record["sha256"] if exists else None,
                    "bundle_sha256": record["sha256"],
                    "matches_bundle": exists,
                }
            )
        return {
            "schema": ablation.SCHEMA,
            "kind": "config-state",
            "state": state,
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "files": files,
        }

    def test_synthetic_slot_passes_integrity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root, checkout, manifest = self.preregister(Path(temporary))
            case_dir = Path(manifest["schedule"][0]["case_root"])
            case_dir.mkdir(parents=True)
            bundle = manifest["bundle"]
            state = self.config_state(bundle, checkout, "default")
            write_json(case_dir / "config_state_before.json", state)
            write_json(case_dir / "config_state_after.json", state)
            write_json(
                case_dir / "config_transition_receipt.json",
                {
                    "schema": ablation.SCHEMA,
                    "kind": "config-transition-receipt",
                    "requested_state": "default",
                    "before": state,
                    "after": state,
                    "bundle": {"manifest_sha256": bundle["manifest_sha256"]},
                },
            )
            server_log = case_dir / "server.log"
            server_log.write_text("synthetic dense FP8 server log\n", encoding="utf-8")
            write_json(
                case_dir / "dense_fp8_warning_gate.json",
                {
                    "status": "ok",
                    "expect": "default",
                    "parsed": {
                        "sha256": ablation.sha256_file(server_log),
                        "fp8_gemm_runner_backend": "triton",
                        "dense_fallbacks": [
                            {"shape": list(shape)} for shape in ablation.EXPECTED_SHAPES
                        ],
                        "dense_tuned_loads": [],
                        "faq_ignored_fused_moe_warning_count": 1,
                        "dense_config_clamp_warning_count": 0,
                    },
                },
            )
            write_json(
                case_dir / "runtime_verification.json",
                {
                    "status": "ok",
                    "failures": [],
                    "checks": {
                        "runtime_hot_experts": ablation.EXPECTED_HOT_EXPERTS,
                        "runtime_mask_layers": 48,
                    },
                },
            )
            model = f"/models/{ablation.EXPECTED_MODEL_SNAPSHOT}"
            stats_path = manifest["inputs"]["frequency_stats"]["runtime_path"]
            write_json(
                case_dir / "case_metadata.json",
                {
                    "mode": "exact",
                    "cpu_threads": 12,
                    "model": model,
                    "frequency_stats": {
                        "path": stats_path,
                        "sha256": manifest["inputs"]["frequency_stats"]["sha256"],
                    },
                    "evaluation_fixture": {
                        "sha256": manifest["inputs"]["evaluation_fixture"]["sha256"]
                    },
                    "ktransformers_commit": ablation.EXPECTED_KT_COMMIT,
                    "sglang_commit": ablation.EXPECTED_SGLANG_COMMIT,
                    "gpu_experts_ratio_cli": "0.77083333333333337",
                    "kv_tokens": 13104,
                },
            )
            outputs = [[index + 1] * 64 for index in range(8)]
            requests = [
                {
                    "success": True,
                    "submitted_input_kind": "input_ids",
                    "server_prompt_token_count_matches_submission": True,
                    "completion_tokens": 64,
                    "output_token_ids": values,
                    "output_token_ids_sha256": ablation.sha256_json(values),
                }
                for values in outputs
            ]
            write_json(
                case_dir / "benchmark.json",
                {
                    "status": "ok",
                    "config": {
                        "concurrencies": [8],
                        "output_tokens_per_request": 64,
                        "require_prompt_token_ids": True,
                        "sampling_params": {
                            "temperature": 0.0,
                            "ignore_eos": True,
                            "sampling_seed": 0,
                        },
                    },
                    "fixture": {
                        "sha256": manifest["inputs"]["evaluation_fixture"]["sha256"],
                        "all_requests_have_prompt_token_ids": True,
                    },
                    "phases": [
                        {
                            "concurrency": 8,
                            "completed_requests": 8,
                            "failed_requests": 0,
                            "expected_output_tokens": 512,
                            "completed_output_tokens": 512,
                            "all_requests_produced_fixed_output": True,
                            "prompt_token_ids_list_sha256": manifest["inputs"]["evaluation_fixture"]["c8_prompt_token_ids_list_sha256"],
                            "output_tokens_per_second": 100.0,
                            "elapsed_seconds": 5.12,
                            "mean_ttft_ms": 1000.0,
                            "p50_ttft_ms": 900.0,
                            "p99_ttft_ms": 1200.0,
                            "mean_tpot_ms": 25.0,
                            "p50_tpot_ms": 24.0,
                            "p99_tpot_ms": 30.0,
                            "requests": requests,
                        }
                    ],
                },
            )
            write_json(
                case_dir / "warmup.json",
                {
                    "status": "ok",
                    "config": {
                        "concurrencies": [8],
                        "output_tokens_per_request": 8,
                        "require_prompt_token_ids": True,
                        "sampling_params": {
                            "temperature": 0.0,
                            "ignore_eos": True,
                            "sampling_seed": 0,
                        },
                    },
                    "fixture": {
                        "sha256": manifest["inputs"]["warmup_fixture"]["sha256"],
                        "all_requests_have_prompt_token_ids": True,
                    },
                    "phases": [
                        {
                            "concurrency": 8,
                            "completed_requests": 8,
                            "failed_requests": 0,
                            "expected_output_tokens": 64,
                            "completed_output_tokens": 64,
                            "all_requests_produced_fixed_output": True,
                            "prompt_token_ids_list_sha256": manifest["inputs"]["warmup_fixture"]["c8_prompt_token_ids_list_sha256"],
                        }
                    ],
                },
            )
            options = {
                "--model-path": model,
                "--context-length": "512",
                "--max-running-requests": "32",
                "--chunked-prefill-size": "4096",
                "--max-prefill-tokens": "4096",
                "--schedule-policy": "fcfs",
                "--stream-interval": "1",
                "--random-seed": "0",
                "--kt-weight-path": model,
                "--kt-cpuinfer": "12",
                "--kt-method": "FP8",
                "--kt-threadpool-count": "1",
                "--kt-numa-nodes": "0",
                "--kt-gpu-experts-ratio": "0.77083333333333337",
                "--kt-expert-placement-strategy": "frequency",
                "--init-expert-location": stats_path,
                "--kt-max-deferred-experts-per-token": "0",
                "--kt-gpu-prefill-token-threshold": "0",
                "--max-total-tokens": "13104",
                "--attention-backend": "flashinfer",
                "--fp8-gemm-backend": "triton",
            }
            command = ["taskset", "-c", "0-13", "/python", "-m", "sglang.launch_server"]
            for flag, value in options.items():
                command.extend((flag, value))
            command.extend(
                (
                    "--trust-remote-code",
                    "--enable-mixed-chunk",
                    "--disable-radix-cache",
                    "--disable-shared-experts-fusion",
                )
            )
            (case_dir / "server_command.txt").write_text(
                shlex.join(command) + "\n", encoding="utf-8"
            )
            (case_dir / "ready_gpu.csv").write_text(
                "2026/08/13 00:00:00, 26000, 0, 100\n", encoding="utf-8"
            )
            (case_dir / "phase_nvml.csv").write_text(
                "2026/08/13 00:00:01, 0, 26100, 70, 10, 200, 2500, 50, 5, 16\n",
                encoding="utf-8",
            )
            (case_dir / "runner.returncode.txt").write_text("0\n", encoding="utf-8")
            output = case_dir / "integrity_gate.json"
            result = ablation.verify_slot(
                argparse.Namespace(
                    case_dir=case_dir,
                    run_manifest=run_root / "run_manifest.json",
                    arm="D",
                    output=output,
                )
            )
            self.assertEqual(result["status"], "ok", result["failures"])
            self.assertEqual(result["metrics"]["phase_peak_gpu_util_percent"], 70)
            self.assertEqual(result["output"]["c8_output_token_ids_sha256"], ablation.sha256_json(outputs))

    def make_summary_root(
        self, root: Path, *, missing_slot: int | None = None, divergent: bool = False,
        zero_metrics: bool = False
    ) -> Path:
        run_root = root / "summary-run"
        schedule = []
        statuses = []
        for slot, arm, state in ablation.SCHEDULE:
            case_root = run_root / "cases/block1" / f"slot{slot}_{arm}"
            item = {
                "slot": slot,
                "arm": arm,
                "config_state": state,
                "case_label": f"block1/slot{slot}_{arm}",
                "case_root": str(case_root),
            }
            schedule.append(item)
            statuses.append(
                {
                    **item,
                    "status": "ok",
                    "returncode": 0,
                    "started_at": "start",
                    "ended_at": "end",
                    "detail": None,
                }
            )
            if slot == missing_slot:
                continue
            value = 0.0 if zero_metrics else float(100 + slot)
            output_hash = "different" if divergent and arm == "T" else "same"
            write_json(
                case_root / "integrity_gate.json",
                {
                    "schema": ablation.SCHEMA,
                    "status": "ok",
                    "arm": arm,
                    "config_state": state,
                    "case_dir": str(case_root.resolve()),
                    "bundle_manifest_sha256": "bundle-sha",
                    "output": {"c8_output_token_ids_sha256": output_hash},
                    "metrics": {
                        "output_tokens_per_second": value,
                        "mean_ttft_ms": value,
                        "mean_tpot_ms": value,
                        "ready_memory_used_mib": value,
                        "phase_peak_memory_used_mib": value,
                    },
                },
            )
        write_json(
            run_root / "run_manifest.json",
            {"bundle": {"manifest_sha256": "bundle-sha"}, "schedule": schedule},
        )
        write_json(
            run_root / "execution_status.json",
            {"schema": ablation.SCHEMA, "slots": statuses},
        )
        return run_root

    def test_summary_rejects_missing_slot_and_output_divergence(self) -> None:
        for name, kwargs in (
            ("missing", {"missing_slot": 4}),
            ("divergent", {"divergent": True}),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run_root = self.make_summary_root(Path(temporary), **kwargs)
                result = ablation.summarize(run_root)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["output_hash_exact_across_d_t"])

    def test_summary_renders_zero_as_zero_not_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.make_summary_root(Path(temporary), zero_metrics=True)
            result = ablation.summarize(run_root)
            self.assertEqual(result["status"], "ok", result["failures"])
            report = (run_root / "DENSE_FP8_DTTD_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("| D | 2 | 0.000000 | 0.000000 | 0.000000 |", report)
            self.assertNotIn("| D | 2 | nan", report)


if __name__ == "__main__":
    unittest.main()
