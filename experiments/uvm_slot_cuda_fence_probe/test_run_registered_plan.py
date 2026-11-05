#!/usr/bin/env python3
"""CPU-only contracts for the synchronous fenced-plan UAPI."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_registered_plan.py")
SPEC = importlib.util.spec_from_file_location("run_registered_plan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


START_GENERATION = 41


def success(
    generation: int, ticket: int, *, owner_cutover: bool
) -> dict[str, object]:
    return {
        "ioctl_command": 96,
        "ioctl_ret": 0,
        "rm_status": 0,
        "rejected": False,
        "generation": generation,
        "new_generation": generation + 1,
        "accepted_bytes": probe.base.BLOCK_BYTES,
        "success_count": probe.base.SLICES_PER_BLOCK,
        "remap_count": probe.base.SLICES_PER_BLOCK,
        "owner_cutover": owner_cutover,
        "fence_accepted": True,
        "completion_signaled": True,
        "completion_before": 0,
        "completion_after_ioctl": ticket,
        "completion_wait_enqueued": True,
        "completion_after_wait": ticket,
        "ticket": ticket,
    }


def rejection(
    generation: int,
    *,
    command: int = 96,
    fence_accepted: bool = False,
) -> dict[str, object]:
    return {
        "ioctl_command": command,
        "ioctl_ret": -1,
        "errno": 22,
        "rm_status": 1,
        "rejected": True,
        "generation": generation,
        "new_generation": generation,
        "accepted_bytes": 0,
        "success_count": 0,
        "remap_count": 0,
        "owner_cutover": False,
        "fence_accepted": fence_accepted,
        "completion_signaled": False,
        "completion_before": 0,
        "completion_after_ioctl": 0,
        "completion_wait_enqueued": False,
        "completion_after_wait": 0,
        "ticket": 1,
    }


def valid_payload(case: str) -> dict[str, object]:
    if case == "correct_ticket":
        return {
            "case": case,
            "generation_start": START_GENERATION,
            "generation_end": START_GENERATION + 2,
            "compute_checksum_completed_before_ticket": True,
            "responses": [
                success(START_GENERATION, 1, owner_cutover=False),
                success(START_GENERATION + 1, 2, owner_cutover=True),
            ],
        }
    if case == "replay_ticket":
        return {
            "case": case,
            "generation_start": START_GENERATION,
            "generation_end": START_GENERATION + 1,
            "responses": [
                success(START_GENERATION, 1, owner_cutover=False),
                rejection(START_GENERATION + 1),
            ],
        }
    if case == "legacy_ioctl93":
        return {
            "case": case,
            "generation_start": START_GENERATION,
            "generation_end": START_GENERATION,
            "responses": [rejection(START_GENERATION, command=93)],
        }
    return {
        "case": case,
        "generation_start": START_GENERATION,
        "generation_end": START_GENERATION,
        "responses": [
            rejection(
                START_GENERATION,
                fence_accepted=case == "failed_execute_no_completion",
            )
        ],
    }


class AbiTests(unittest.TestCase):
    def test_command_96_layout(self) -> None:
        probe.assert_execute_abi()
        self.assertEqual(probe.ctypes.sizeof(probe.ExecuteFencedParams), 128)
        self.assertEqual(probe.ExecuteFencedParams.fence_flags.offset, 84)
        self.assertEqual(
            probe.ExecuteFencedParams.compute_semaphore_address.offset, 88
        )
        self.assertEqual(probe.ExecuteFencedParams.rmStatus.offset, 124)

class ContractTests(unittest.TestCase):
    def test_all_six_cases_are_accepted(self) -> None:
        self.assertEqual(len(probe.CASES), 6)
        for case in probe.CASES:
            with self.subTest(case=case):
                probe.require_case_contract(case, valid_payload(case))

    def test_replayed_ticket_cannot_be_accepted(self) -> None:
        payload = valid_payload("replay_ticket")
        payload["responses"][1]["fence_accepted"] = True  # type: ignore[index]
        with self.assertRaisesRegex(probe.base.ProbeError, "stale ticket was accepted"):
            probe.require_case_contract("replay_ticket", payload)

    def test_wrong_pool_va_cannot_signal_completion(self) -> None:
        payload = valid_payload("wrong_pool_va")
        payload["responses"][0]["completion_signaled"] = True  # type: ignore[index]
        with self.assertRaisesRegex(probe.base.ProbeError, "signaled completion"):
            probe.require_case_contract("wrong_pool_va", payload)

    def test_same_address_cannot_be_accepted(self) -> None:
        payload = valid_payload("same_address")
        payload["responses"][0]["fence_accepted"] = True  # type: ignore[index]
        with self.assertRaisesRegex(probe.base.ProbeError, "contract was accepted"):
            probe.require_case_contract("same_address", payload)

    def test_legacy_case_must_really_use_ioctl_93(self) -> None:
        payload = valid_payload("legacy_ioctl93")
        payload["responses"][0]["ioctl_command"] = 96  # type: ignore[index]
        with self.assertRaisesRegex(probe.base.ProbeError, "command 93 was not exercised"):
            probe.require_case_contract("legacy_ioctl93", payload)

    def test_execute_failure_requires_accepted_compute_fence(self) -> None:
        payload = valid_payload("failed_execute_no_completion")
        payload["responses"][0]["fence_accepted"] = False  # type: ignore[index]
        with self.assertRaisesRegex(probe.base.ProbeError, "compute fence was not accepted"):
            probe.require_case_contract("failed_execute_no_completion", payload)

    def test_execute_failure_must_not_signal_completion(self) -> None:
        payload = valid_payload("failed_execute_no_completion")
        payload["responses"][0]["completion_after_ioctl"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(probe.base.ProbeError, "changed completion cell"):
            probe.require_case_contract("failed_execute_no_completion", payload)

    def test_rejection_must_not_publish_owner(self) -> None:
        for case in (
            "replay_ticket",
            "wrong_pool_va",
            "same_address",
            "legacy_ioctl93",
            "failed_execute_no_completion",
        ):
            payload = valid_payload(case)
            index = 1 if case == "replay_ticket" else 0
            payload["responses"][index]["owner_cutover"] = True  # type: ignore[index]
            with self.subTest(case=case), self.assertRaisesRegex(
                probe.base.ProbeError, "owner was published"
            ):
                probe.require_case_contract(case, payload)

    def test_rejection_must_not_advance_generation(self) -> None:
        payload = valid_payload("wrong_pool_va")
        mutated = copy.deepcopy(payload)
        mutated["responses"][0]["new_generation"] = START_GENERATION + 1  # type: ignore[index]
        with self.assertRaisesRegex(probe.base.ProbeError, "generation changed"):
            probe.require_case_contract("wrong_pool_va", mutated)

    def test_success_requires_stream_wait(self) -> None:
        payload = valid_payload("correct_ticket")
        payload["responses"][1]["completion_wait_enqueued"] = False  # type: ignore[index]
        with self.assertRaisesRegex(probe.base.ProbeError, "stream completion wait"):
            probe.require_case_contract("correct_ticket", payload)


if __name__ == "__main__":
    unittest.main()
