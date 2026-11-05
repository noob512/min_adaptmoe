#!/usr/bin/env python3
"""CPU-only tests for the common-trace ABBA control plane."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("common_trace_abba.py")
SPEC = importlib.util.spec_from_file_location("common_trace_abba", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CommonTraceAbbaTest(unittest.TestCase):
    def test_alternating_abba_schedule(self) -> None:
        self.assertEqual(MODULE.expected_sequence(1), ["A", "B", "B", "A"])
        self.assertEqual(MODULE.expected_sequence(2), ["B", "A", "A", "B"])
        self.assertEqual(MODULE.expected_sequence(3), ["A", "B", "B", "A"])
        with self.assertRaises(ValueError):
            MODULE.expected_sequence(0)

    def test_cluster_bootstrap_uses_complete_blocks(self) -> None:
        blocks = [
            {"kt": [10.0, 10.0], "custom": [2.0, 2.0]},
            {"kt": [12.0, 12.0], "custom": [3.0, 3.0]},
            {"kt": [8.0, 8.0], "custom": [2.0, 2.0]},
        ]
        first = MODULE.cluster_bootstrap_ratio(blocks, samples=1_000, seed=7)
        second = MODULE.cluster_bootstrap_ratio(blocks, samples=1_000, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lower_95"], 10.0 / (7.0 / 3.0))
        self.assertGreaterEqual(first["upper_95"], 10.0 / (7.0 / 3.0))
        with self.assertRaises(ValueError):
            MODULE.cluster_bootstrap_ratio([{"kt": [1.0], "custom": []}])

    def test_metric_summary_keeps_process_values(self) -> None:
        result = MODULE.metric_summary([1.0, 2.0, 3.0])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["values"], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["cv"], 0.5)


if __name__ == "__main__":
    unittest.main()
