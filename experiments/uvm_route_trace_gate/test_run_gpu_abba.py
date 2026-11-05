#!/usr/bin/env python3
"""Pure-userspace tests for the route-trace GPU A/B controller."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_gpu_abba.py")
SPEC = importlib.util.spec_from_file_location("uvm_route_trace_abba", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ABBA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ABBA)


class RouteTraceAbbaTest(unittest.TestCase):
    def test_parse_and_delta(self) -> None:
        before = ABBA.parse_route_stats(
            """enabled 0
debug_emit_calls 10
disabled_race_drops 0
always_emit_calls 0
timestamp_calls 10
format_calls 10
mutex_acquires 10
records 10
bytes 1000
"""
        )
        after = dict(before)
        after["enabled"] = 1
        for name in (
            "debug_emit_calls",
            "timestamp_calls",
            "format_calls",
            "mutex_acquires",
            "records",
        ):
            after[name] += 7
        after["bytes"] += 700
        delta = ABBA.counter_delta(before, after)
        self.assertNotIn("enabled", delta)
        self.assertEqual(delta["debug_emit_calls"], 7)
        self.assertEqual(delta["bytes"], 700)

    def test_route_delta_contract(self) -> None:
        off = {name: 0 for name in ABBA.ROUTE_ACTIVITY_COUNTERS}
        ABBA.validate_route_delta(False, off)

        on = dict(off)
        for name in (
            "debug_emit_calls",
            "timestamp_calls",
            "format_calls",
            "mutex_acquires",
            "records",
        ):
            on[name] = 17
        on["bytes"] = 2048
        ABBA.validate_route_delta(True, on)

        broken = dict(on)
        broken["mutex_acquires"] -= 1
        with self.assertRaisesRegex(RuntimeError, "mutex_acquires"):
            ABBA.validate_route_delta(True, broken)

    def test_probe_payload_and_comparison(self) -> None:
        blocks = 2
        entries = blocks * 4
        logical_bytes = blocks * 2 * ABBA.MIB

        def phase(mode: int, wall_ns: int) -> dict[str, object]:
            return {
                "ok": True,
                "mode": mode,
                "requested_count": entries,
                "success_count": entries,
                "failure_count": 0,
                "requested_bytes": logical_bytes,
                "accepted_bytes": logical_bytes,
                "remap_count": entries,
                "ioctl_calls": 1,
                "ioctl_wall_ns": wall_ns,
            }

        payload = {
            "status": "pass",
            "module_srcversion": "ABC",
            "blocks": blocks,
            "entries": entries,
            "logical_bytes": logical_bytes,
            "restore_a": phase(21, 10),
            "transition_b": phase(20, 20),
        }
        metrics = ABBA.validate_probe_payload(
            payload,
            blocks=blocks,
            expected_srcversion="ABC",
        )
        self.assertEqual(metrics["hot_ioctl_wall_ns"], 30)

        arms = [
            {"trace_enabled": False, "metrics": metrics},
            {
                "trace_enabled": True,
                "metrics": {
                    "restore_a_ioctl_wall_ns": 20,
                    "transition_b_ioctl_wall_ns": 40,
                    "hot_ioctl_wall_ns": 60,
                },
            },
        ]
        result = ABBA.comparison(arms)
        self.assertEqual(result["hot_ioctl_wall_ns"]["on_over_off"], 2.0)


if __name__ == "__main__":
    unittest.main()
