#!/usr/bin/env python3
"""Exercise generation-free registered-slot teardown against the real driver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_probe import (  # type: ignore[import-not-found]
    BLOCK_BYTES,
    DEFAULT_LIBRARY,
    DEFAULT_STATS,
    GpuBackend,
    INITIAL_PLAN_ID,
    MODE_INITIAL,
    ProbeError,
    SLICES_PER_BLOCK,
    build_entries,
    require,
    require_preallocate,
    stats_delta,
)


def _require_abort(
    result: dict[str, object],
    *,
    generation: int,
    plans: int,
    slots: int,
    bytes_: int,
) -> None:
    expected = {
        "generation_before": generation,
        "destroyed_plan_count": plans,
        "released_slot_count": slots,
        "released_bytes": bytes_,
        "remaining_slot_count": 0,
        "remaining_bytes": 0,
        "ioctl_calls": 1,
        "rm_status": 0,
    }
    require(result.get("ok") is True, f"session abort failed: {result}")
    require(result.get("session_cleared") is True, "session was not cleared")
    require(result.get("poisoned") is False, "cleared session stayed poisoned")
    for name, value in expected.items():
        require(int(result.get(name, -1)) == value, f"abort {name} mismatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    backend = GpuBackend(
        library=args.library,
        stats=args.stats,
        device=args.device,
    )
    payload: dict[str, Any] = {"status": "failed"}
    try:
        dataset = backend.create_dataset(0, 0x51A60B17, 1)
        pointers = backend.dataset_pointers(dataset, 1)
        entries = build_entries(pointers, layer_id=0, reuse_required=False)
        require(len(entries) == SLICES_PER_BLOCK, "unexpected one-block geometry")
        before = backend.read_stats()

        first_preallocate = backend.preallocate(entries)
        require_preallocate(
            first_preallocate,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
        )
        first_register = backend.register_plan_raw(
            INITIAL_PLAN_ID,
            0,
            entries,
            mode=MODE_INITIAL,
        )
        require(first_register.get("ok") is True, "raw bootstrap register failed")
        first_generation = int(first_register["generation"])
        require(first_generation > 0, "first generation is not positive")

        # This raw path deliberately bypasses the Python plan/generation cache.
        first_execute = backend.execute_plan_raw(
            INITIAL_PLAN_ID,
            0,
            first_generation,
        )
        require(first_execute.get("rejected") is False, "raw execute was rejected")
        require(int(first_execute.get("rm_status", -1)) == 0, "raw execute failed")
        executed_generation = int(first_execute["new_generation"])
        require(
            executed_generation == first_generation + 1,
            "raw execute generation did not advance",
        )
        actual, expected_checksum = backend.checksum(dataset)
        require(actual == expected_checksum, "slot-mapped checksum mismatch")

        first_abort = backend.uvm.driver_slot_abort_session(device=args.device)
        _require_abort(
            first_abort,
            generation=executed_generation,
            plans=1,
            slots=SLICES_PER_BLOCK,
            bytes_=BLOCK_BYTES,
        )
        first_state = backend.state_snapshot()
        require(first_state["allocated_slots"] == 0, "first abort left local slots")
        require(first_state["plan_count"] == 0, "first abort left local plans")
        require(first_state["tracked_generation"] is None, "generation cache survived")
        require(first_state["session_open"] is True, "abort closed the recovery fd")
        first_close = backend.close()
        require(first_close.get("ok") is True, "first fd close failed")

        # Reuse the same managed VA in a new logical session. The module-wide
        # generation counter must prevent an ABA match with the old plan.
        second_preallocate = backend.preallocate(entries)
        require_preallocate(
            second_preallocate,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
        )
        second_register = backend.register_plan_raw(
            INITIAL_PLAN_ID,
            0,
            entries,
            mode=MODE_INITIAL,
        )
        require(second_register.get("ok") is True, "second bootstrap failed")
        second_generation = int(second_register["generation"])
        require(
            second_generation > executed_generation,
            "generation did not remain monotonic across abort/reopen",
        )
        second_abort = backend.uvm.driver_slot_abort_session(device=args.device)
        _require_abort(
            second_abort,
            generation=second_generation,
            plans=1,
            slots=SLICES_PER_BLOCK,
            bytes_=BLOCK_BYTES,
        )
        second_close = backend.close()
        require(second_close.get("ok") is True, "second fd close failed")

        after = backend.read_stats()
        delta = stats_delta(before, after)
        expected_counters = {
            "slot_session_abort_calls": 2,
            "slot_session_abort_successes": 2,
            "slot_session_abort_failures": 0,
            "slot_session_abort_released_slots": 2 * SLICES_PER_BLOCK,
            "slot_session_abort_released_bytes": 2 * BLOCK_BYTES,
            "slot_session_abort_remaining_slots": 0,
        }
        for name, value in expected_counters.items():
            require(name in delta, f"missing abort counter {name}")
            require(int(delta[name]) == value, f"counter {name} mismatch")

        payload = {
            "status": "pass",
            "unknown_generation_recovered": True,
            "pte_backing_released": True,
            "generation_monotonic": True,
            "first_generation": first_generation,
            "executed_generation": executed_generation,
            "second_generation": second_generation,
            "first_abort": first_abort,
            "second_abort": second_abort,
            "first_close": first_close,
            "second_close": second_close,
            "counter_delta": {
                name: int(delta[name]) for name in expected_counters
            },
        }
    except Exception as exc:
        payload["error"] = repr(exc)
        payload["error_type"] = type(exc).__name__
    finally:
        finish = backend.finish()
        payload["finish"] = finish
        if not bool(finish.get("ok")):
            payload["status"] = "failed"
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    try:
        result = run(args)
    except ProbeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
