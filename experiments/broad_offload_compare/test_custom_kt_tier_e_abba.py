#!/usr/bin/env python3
"""CPU-only tests for the custom-vs-KT Tier-E controller."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import custom_kt_tier_e_abba as target


class ScheduleTests(unittest.TestCase):
    def test_schedule_is_three_balanced_blocks_per_point(self) -> None:
        root = Path("/tmp/frozen-custom-kt")
        schedule = target.build_schedule(root)
        self.assertEqual(len(schedule), 36)
        for concurrency in target.CONCURRENCIES:
            point = [item for item in schedule if item["concurrency"] == concurrency]
            self.assertEqual(sum(item["arm"] == "A" for item in point), 6)
            self.assertEqual(sum(item["arm"] == "B" for item in point), 6)
            for block in range(1, 4):
                sequence = [
                    item["arm"] for item in point if item["block"] == block
                ]
                expected = (
                    list(target.ODD_SEQUENCE)
                    if block % 2
                    else list(target.EVEN_SEQUENCE)
                )
                self.assertEqual(sequence, expected)

    def test_cluster_bootstrap_resamples_complete_blocks(self) -> None:
        blocks = [
            {"A": [10.0, 10.0], "B": [5.0, 5.0]},
            {"A": [20.0, 20.0], "B": [10.0, 10.0]},
            {"A": [30.0, 30.0], "B": [15.0, 15.0]},
        ]
        result = target.cluster_bootstrap(blocks, seed=7, draws=500)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["cluster_unit"], "complete four-slot ABBA/BAAB block")
        self.assertEqual(result["complete_clusters"], 3)
        self.assertAlmostEqual(result["ratio"], 2.0)
        self.assertEqual(result["ci95_percentile"], [2.0, 2.0])
        self.assertIsNone(
            target.cluster_bootstrap([{"A": [1.0], "B": [1.0, 1.0]}], seed=1)
        )


class AtomicStatusTests(unittest.TestCase):
    def test_status_transitions_are_atomic_and_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            schedule = target.build_schedule(root)
            schedule_doc = {
                "schema": target.SCHEMA,
                "kind": "frozen-schedule",
                "slots": schedule,
            }
            target.atomic_json(root / "schedule.json", schedule_doc)
            schedule_sha = target.sha256_file(root / "schedule.json")
            target.atomic_json(
                root / "run_manifest.json",
                {
                    "schema": target.SCHEMA,
                    "kind": "preregistered-run",
                    "schedule": {"sha256": schedule_sha},
                },
            )
            target.atomic_json(
                root / "execution_status.json",
                {
                    "schema": target.SCHEMA,
                    "kind": "execution-status",
                    "schedule_sha256": schedule_sha,
                    "slots": [
                        {
                            "slot_id": item["slot_id"],
                            "index": item["index"],
                            "status": "pending",
                            "returncode": None,
                            "started_at": None,
                            "ended_at": None,
                            "detail": "pending",
                            "history": [],
                        }
                        for item in schedule
                    ],
                },
            )
            slot_id = schedule[0]["slot_id"]
            running = target.update_status(
                argparse.Namespace(
                    run_root=root,
                    slot_id=slot_id,
                    status="running",
                    returncode=0,
                    detail="started",
                )
            )
            self.assertEqual(running["status"], "running")
            success = target.update_status(
                argparse.Namespace(
                    run_root=root,
                    slot_id=slot_id,
                    status="success",
                    returncode=0,
                    detail="passed",
                )
            )
            self.assertEqual(success["status"], "success")
            self.assertEqual(len(success["history"]), 2)
            with self.assertRaises(ValueError):
                target.update_status(
                    argparse.Namespace(
                        run_root=root,
                        slot_id=slot_id,
                        status="failed",
                        returncode=1,
                        detail="must not overwrite terminal evidence",
                    )
                )


class FailClosedTests(unittest.TestCase):
    def test_incomplete_bundle_is_rejected_without_fragment_scan(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "wait for tuning/bundle completion"):
                target.validate_dense_bundle(Path(raw))

    def test_missing_cross_framework_quality_is_a_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "missing.json"
            result = target.validate_quality_gate(
                path,
                {
                    "inputs": {
                        "evaluation_fixture": {
                            "sha256": "0" * 64,
                            "prompt_token_ids_list_sha256": "1" * 64,
                        }
                    }
                },
            )
            self.assertEqual(result["status"], "missing")
            self.assertFalse(result["pass"])

    def test_validator_exceptions_fail_closed(self) -> None:
        result = target.validate_slot(
            {
                "slot_id": "c1-b1-s1-A",
                "arm": "A",
                "framework": "custom",
                "artifact_root": "/definitely/missing",
            },
            {"inputs": {"custom_plan": {"snapshot": "/definitely/missing"}}},
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("fail-closed validator exception", result["failures"][0])

    def test_controller_has_no_torch_or_cuda_runtime_import(self) -> None:
        source = Path(target.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("nvidia-smi", source)


if __name__ == "__main__":
    unittest.main()
