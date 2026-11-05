#!/usr/bin/env python3
"""Pure-userspace tests for the registered-plan probe state machine."""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).with_name("run_probe.py")
SPEC = importlib.util.spec_from_file_location("registered_plan_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class FakeBackend:
    def __init__(self, *, initial_generation: int = 37) -> None:
        names = set(probe.CORE_ACTIVITY_COUNTERS) | set(
            probe.CONTROL_PLANE_COUNTERS
        ) | {
            "slot_direct_per_block_calls",
            "slot_direct_shared_calls",
            "slot_direct_pte_4k_blocks",
            "slot_direct_pte_64k_blocks",
            "slot_direct_pte_2m_blocks",
            "slot_direct_skipped_unmap_blocks",
            "slot_direct_stage_sync_calls",
            "slot_direct_tracker_waits",
            "slot_direct_vidmem_allocations",
            "slot_direct_vidmem_evictions",
            "slot_direct_transition_group_clear_fallbacks",
            "slot_direct_transition_clear_ranges",
            "slot_direct_transition_clear_unique_blocks",
        }
        self.stats = {name: 0 for name in names}
        self.pointers: dict[int, tuple[int, ...]] = {}
        self.dataset_layers: dict[int, int] = {}
        self.next_dataset = 0
        self.plans: dict[int, dict[str, Any]] = {}
        self.generation: int | None = None
        self.next_generation = initial_generation
        self.bootstrap_registered = False
        self.canonical_geometry: dict[tuple[int, int], tuple[int, int]] | None = None
        self.transition_layers: set[int] = set()
        self.invalidated_plan_ids: set[int] = set()
        self.backing_invalidated_plan_ids: set[int] = set()
        self.gpu_uuid = bytes(range(16))
        self.active_layer: int | None = None
        self.owners = 0
        self.session_open = False
        self.calls: list[tuple[Any, ...]] = []

    def create_dataset(self, layer_id: int, seed: int, blocks: int) -> object:
        base = 0x1_0000_0000 + layer_id * 0x1000_0000
        handle = self.next_dataset
        self.next_dataset += 1
        self.pointers[handle] = tuple(
            base + index * probe.BLOCK_BYTES for index in range(blocks)
        )
        self.dataset_layers[handle] = layer_id
        self.calls.append(("create", layer_id, seed, blocks))
        return handle

    def dataset_pointers(self, dataset: object, blocks: int) -> tuple[int, ...]:
        return self.pointers[int(dataset)][:blocks]

    def checksum(self, dataset: object) -> tuple[int, int]:
        layer_id = int(dataset)
        if self.active_layer != layer_id:
            raise AssertionError("probe checksummed an inactive fake layer")
        value = 0xABC00000 + layer_id
        self.calls.append(("checksum", layer_id))
        return value, value

    def read_stats(self) -> dict[str, int]:
        return dict(self.stats)

    def preallocate(self, entries):
        count = len(entries)
        logical_bytes = count * probe.SLICE_BYTES
        self.stats["slot_legacy_ioctl_calls"] += 1
        self.owners = count
        self.session_open = True
        self.calls.append(("preallocate", count))
        return {
            "ok": True,
            "mode": probe.MODE_PREALLOC,
            "requested_count": count,
            "success_count": count,
            "failure_count": 0,
            "requested_bytes": logical_bytes,
            "accepted_bytes": logical_bytes,
            "remap_count": 0,
            "ioctl_calls": 1,
            "entry_statuses": [0] * count,
        }

    def register_plan(
        self, plan_id, layer_id, entries, *, transition
    ) -> dict[str, object]:
        if self.generation is None:
            self.generation = self.next_generation
            self.next_generation += 1
        mode = probe.MODE_TRANSITION if transition else probe.MODE_INITIAL
        copied = tuple(tuple(record) for record in entries)
        geometry = {
            (int(record[2]), int(record[4])): (int(record[1]), int(record[5]))
            for record in copied
        }
        if not transition:
            assert not self.bootstrap_registered
            self.bootstrap_registered = True
            self.canonical_geometry = geometry
        else:
            assert self.bootstrap_registered
            assert geometry == self.canonical_geometry
            assert layer_id not in self.transition_layers
            self.transition_layers.add(layer_id)
        self.plans[plan_id] = {
            "layer_id": layer_id,
            "mode": mode,
            "entries": copied,
        }
        self.stats["slot_plan_register_calls"] += 1
        self.calls.append(("register", plan_id, layer_id, mode, len(copied)))
        return {
            "ok": True,
            "reason": "ok",
            "plan_id": plan_id,
            "layer_id": layer_id,
            "mode": mode,
            "generation": self.generation,
            "requested_count": len(copied),
            "requested_bytes": len(copied) * probe.SLICE_BYTES,
            "ioctl_ret": 0,
            "rm_status": 0,
        }

    def register_plan_raw(
        self,
        plan_id,
        layer_id,
        entries,
        *,
        mode,
        count_override=None,
        uuid_mismatch=False,
    ) -> dict[str, object]:
        copied = tuple(tuple(map(int, record)) for record in entries)
        count = len(copied) if count_override is None else int(count_override)
        submitted = copied[:count]
        geometry = {
            (record[2], record[4]): (record[1], record[5])
            for record in submitted
        }
        rejected = False
        if mode == probe.MODE_INITIAL:
            rejected = self.bootstrap_registered
        elif mode == probe.MODE_TRANSITION:
            rejected = (
                not self.bootstrap_registered
                or uuid_mismatch
                or count != len(self.canonical_geometry or {})
                or geometry != self.canonical_geometry
                or layer_id in self.transition_layers
            )
        else:
            rejected = True
        self.stats["slot_plan_register_calls"] += 1
        self.calls.append(
            (
                "raw_register",
                plan_id,
                layer_id,
                mode,
                count,
                uuid_mismatch,
                rejected,
            )
        )
        if not rejected:
            self.plans[plan_id] = {
                "layer_id": layer_id,
                "mode": mode,
                "entries": submitted,
            }
            if mode == probe.MODE_INITIAL:
                self.bootstrap_registered = True
                self.canonical_geometry = geometry
            else:
                self.transition_layers.add(layer_id)
        return {
            "ok": not rejected,
            "reason": "rejected" if rejected else "unexpected_accept",
            "rejected": rejected,
            "plan_id": plan_id,
            "layer_id": layer_id,
            "mode": mode,
            "generation": 0 if rejected else self.generation,
            "requested_count": count,
            "requested_bytes": 0 if rejected else sum(record[1] for record in submitted),
            "ioctl_ret": -1 if rejected else 0,
            "errno": 22 if rejected else 0,
            "rm_status": 1 if rejected else 0,
            "ioctl_calls": 1,
        }

    def execute_plan(self, plan_id, layer_id, generation) -> dict[str, object]:
        plan = self.plans[plan_id]
        assert plan["layer_id"] == layer_id
        assert self.generation == generation
        entries = len(plan["entries"])
        blocks = entries // probe.SLICES_PER_BLOCK
        logical_bytes = entries * probe.SLICE_BYTES
        transition = plan["mode"] == probe.MODE_TRANSITION
        self.stats["slot_plan_execute_calls"] += 1
        self.stats.update(
            {
                "slot_direct_ioctl_calls": self.stats["slot_direct_ioctl_calls"] + 1,
                "slot_direct_successes": self.stats["slot_direct_successes"] + 1,
                "slot_direct_requested_ranges": self.stats[
                    "slot_direct_requested_ranges"
                ]
                + entries,
                "slot_direct_completed_ranges": self.stats[
                    "slot_direct_completed_ranges"
                ]
                + entries,
                "slot_direct_requested_bytes": self.stats[
                    "slot_direct_requested_bytes"
                ]
                + logical_bytes,
                "slot_direct_completed_bytes": self.stats[
                    "slot_direct_completed_bytes"
                ]
                + logical_bytes,
                "slot_direct_blocks": self.stats["slot_direct_blocks"] + blocks,
                "slot_direct_memcopy_methods": self.stats[
                    "slot_direct_memcopy_methods"
                ]
                + entries,
                "slot_direct_copy_pushes": self.stats["slot_direct_copy_pushes"] + 1,
                "slot_direct_pte_map_ranges": self.stats[
                    "slot_direct_pte_map_ranges"
                ]
                + entries,
                "slot_direct_pte_map_bytes": self.stats[
                    "slot_direct_pte_map_bytes"
                ]
                + logical_bytes,
                "slot_direct_pte_write_entries": self.stats[
                    "slot_direct_pte_write_entries"
                ]
                + blocks * probe.PTE_PER_BLOCK,
                "slot_direct_pte_64k_blocks": self.stats[
                    "slot_direct_pte_64k_blocks"
                ]
                + blocks,
                "slot_direct_transition_fast_index_calls": self.stats[
                    "slot_direct_transition_fast_index_calls"
                ]
                + int(transition),
                "slot_direct_transition_group_clear_calls": self.stats[
                    "slot_direct_transition_group_clear_calls"
                ]
                + int(transition),
                "slot_direct_transition_direct_cutover_calls": self.stats[
                    "slot_direct_transition_direct_cutover_calls"
                ]
                + int(transition),
                "slot_direct_transition_direct_cutover_pages": self.stats[
                    "slot_direct_transition_direct_cutover_pages"
                ]
                + (logical_bytes // 4096 if transition else 0),
            }
        )
        self.generation += 1
        self.next_generation = max(self.next_generation, self.generation + 1)
        self.active_layer = layer_id
        self.calls.append(("execute", plan_id, layer_id, generation))
        return {
            "ok": True,
            "reason": "ok",
            "plan_id": plan_id,
            "layer_id": layer_id,
            "generation": generation,
            "new_generation": self.generation,
            "mode": plan["mode"],
            "requested_count": entries,
            "success_count": entries,
            "failure_count": 0,
            "conflict_count": 0,
            "requested_bytes": logical_bytes,
            "accepted_bytes": logical_bytes,
            "remap_count": entries,
            "tlb_invalidate_count": blocks,
            "copy_push_count": 1,
            "owner_cutover": transition,
            "owner_cutover_count": entries if transition else 0,
            "owner_cutover_trusted": True,
            "owner_state_indeterminate": False,
            "poisoned": False,
            "ioctl_calls": 1,
            "ioctl_ret": 0,
            "rm_status": 0,
            "plan_execute": True,
        }

    def _execute_plan_raw(self, plan_id, layer_id, generation, *, call_name):
        self.stats["slot_plan_execute_calls"] += 1
        identity_invalid = plan_id in self.invalidated_plan_ids
        backing_invalid = plan_id in self.backing_invalidated_plan_ids
        rejected = self.generation != generation or identity_invalid or backing_invalid
        plan = self.plans.get(plan_id)
        preflight_invalid = identity_invalid or backing_invalid
        entries = len(plan["entries"]) if preflight_invalid and plan is not None else 0
        requested_bytes = entries * probe.SLICE_BYTES
        if preflight_invalid:
            self.stats["slot_direct_ioctl_calls"] += 1
            self.stats["slot_direct_shared_calls"] += 1
            self.stats["slot_direct_errors"] += 1
            self.stats["slot_direct_requested_ranges"] += entries
            self.stats["slot_direct_requested_bytes"] += requested_bytes
        if backing_invalid:
            self.stats["slot_direct_transition_fast_index_calls"] += 1
        self.calls.append((call_name, plan_id, layer_id, generation))
        return {
            "ok": False,
            "reason": "expected_negative_rejection",
            "rejected": rejected,
            "plan_id": plan_id,
            "layer_id": layer_id,
            "generation": generation,
            "new_generation": self.generation,
            "requested_bytes": requested_bytes,
            "accepted_bytes": 0,
            "mode": plan["mode"] if preflight_invalid and plan is not None else 0,
            "requested_count": entries,
            "success_count": 0,
            "failure_count": entries,
            "conflict_count": 0,
            "remap_count": 0,
            "tlb_invalidate_count": 0,
            "copy_push_count": 0,
            "owner_cutover": 0,
            "poisoned": False,
            "ioctl_ret": -1 if rejected else 0,
            "errno": 22 if rejected else 0,
            "rm_status": 1 if rejected else 0,
        }

    def execute_stale_raw(self, plan_id, layer_id, generation):
        return self._execute_plan_raw(
            plan_id, layer_id, generation, call_name="stale"
        )

    def execute_plan_raw(self, plan_id, layer_id, generation):
        return self._execute_plan_raw(
            plan_id, layer_id, generation, call_name="raw_execute"
        )

    def legacy_transition_raw(self, entries):
        self.stats["slot_legacy_ioctl_calls"] += 1
        self.stats["slot_legacy_transition_calls"] += 1
        rejected = bool(self.plans)
        self.calls.append(("legacy_transition", len(entries), rejected))
        return {
            "ok": not rejected,
            "reason": "rejected" if rejected else "unexpected_accept",
            "rejected": rejected,
            "mode": probe.MODE_TRANSITION,
            "requested_count": len(entries),
            "success_count": 0,
            "failure_count": 0,
            "conflict_count": 0,
            "requested_bytes": 0,
            "accepted_bytes": 0,
            "remap_count": 0,
            "tlb_invalidate_count": 0,
            "copy_push_count": 0,
            "owner_cutover": False,
            "ioctl_ret": -1 if rejected else 0,
            "errno": 16 if rejected else 0,
            "rm_status": 1 if rejected else 0,
            "ioctl_calls": 1,
        }

    def destroy_plan(self, plan_id, generation):
        assert self.generation == generation
        self.stats["slot_plan_destroy_calls"] += 1
        del self.plans[plan_id]
        self.invalidated_plan_ids.discard(plan_id)
        self.backing_invalidated_plan_ids.discard(plan_id)
        current = generation if self.plans else 0
        if not self.plans:
            self.generation = None
            self.bootstrap_registered = False
            self.canonical_geometry = None
            self.transition_layers.clear()
        self.calls.append(("destroy", plan_id, generation))
        return {
            "ok": True,
            "reason": "ok",
            "plan_id": plan_id,
            "generation": generation,
            "current_generation": current,
            "destroyed": True,
            "ioctl_ret": 0,
            "rm_status": 0,
        }

    def close(self):
        count = self.owners
        self.stats["slot_legacy_ioctl_calls"] += 1
        self.owners = 0
        self.session_open = False
        self.calls.append(("close", count))
        return {
            "ok": True,
            "reason": "closed",
            "closed": True,
            "plan_destroy": [],
            "release": {
                "ok": True,
                "requested_count": count,
                "success_count": count,
                "failure_count": 0,
                "accepted_bytes": count * probe.SLICE_BYTES,
            },
        }

    def release_ranges(self, entries):
        count = len(entries)
        logical_bytes = count * probe.SLICE_BYTES
        assert self.owners == count
        self.stats["slot_legacy_ioctl_calls"] += 1
        self.owners = 0
        self.backing_invalidated_plan_ids.update(self.plans)
        self.calls.append(("release", count))
        return {
            "ok": True,
            "mode": 19,
            "requested_count": count,
            "success_count": count,
            "failure_count": 0,
            "requested_bytes": logical_bytes,
            "accepted_bytes": logical_bytes,
            "fd_candidate_ioctl_calls": 1,
        }

    def destroy_dataset(self, dataset: object) -> None:
        handle = int(dataset)
        destroyed_vas = self.pointers.pop(handle)
        self.dataset_layers.pop(handle)
        for plan_id, plan in self.plans.items():
            if any(
                any(
                    int(record[0]) <= va < int(record[0]) + int(record[1])
                    for va in destroyed_vas
                )
                for record in plan["entries"]
            ):
                self.invalidated_plan_ids.add(plan_id)
        self.calls.append(("destroy_dataset", handle))

    def state_snapshot(self):
        return {
            "allocated_slots": self.owners,
            "plan_count": len(self.plans),
            "tracked_generation": self.generation,
            "session_open": self.session_open,
        }


class StaleSideEffectBackend(FakeBackend):
    def execute_stale_raw(self, plan_id, layer_id, generation):
        result = super().execute_stale_raw(plan_id, layer_id, generation)
        self.stats["slot_direct_copy_pushes"] += 1
        return result


class MissingStaleExecuteCounterBackend(FakeBackend):
    def execute_stale_raw(self, plan_id, layer_id, generation):
        result = super().execute_stale_raw(plan_id, layer_id, generation)
        self.stats["slot_plan_execute_calls"] -= 1
        return result


class VaMutationBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.mutated = False

    def execute_plan(self, plan_id, layer_id, generation):
        result = super().execute_plan(plan_id, layer_id, generation)
        self.mutated = True
        return result

    def dataset_pointers(self, dataset: object, blocks: int) -> tuple[int, ...]:
        values = super().dataset_pointers(dataset, blocks)
        if self.mutated and int(dataset) == 0:
            return (values[0] + 4096,) + values[1:]
        return values


class CloseLeakBackend(FakeBackend):
    def close(self):
        result = super().close()
        self.owners = 1
        return result


class LegacyTransitionLeakBackend(FakeBackend):
    def close(self):
        result = super().close()
        self.stats["slot_legacy_transition_calls"] += 1
        return result


class ProbeStateMachineTests(unittest.TestCase):
    def test_happy_path_proves_generation_checksum_and_cleanup(self) -> None:
        backend = FakeBackend()
        result = probe.exercise_registered_plans(backend, blocks=1)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["execution_order"], [0, 1, 2, 3, 0])
        generation = result["generation_start"]
        self.assertEqual(generation, 37)
        self.assertEqual(result["generation_end"], generation + 5)
        self.assertTrue(result["zero_slots_after_close"])
        self.assertEqual(
            result["lifecycle_control_delta"],
            {
                "slot_plan_register_calls": 5,
                "slot_plan_execute_calls": 6,
                "slot_plan_destroy_calls": 5,
                "slot_legacy_ioctl_calls": 2,
                "slot_legacy_transition_calls": 0,
            },
        )
        self.assertEqual(
            [call for call in backend.calls if call[0] == "execute"],
            [
                ("execute", 1, 0, generation),
                ("execute", 6, 1, generation + 1),
                ("execute", 10, 2, generation + 2),
                ("execute", 14, 3, generation + 3),
                ("execute", 2, 0, generation + 4),
            ],
        )
        self.assertEqual(
            [call for call in backend.calls if call[0] == "stale"],
            [("stale", 6, 1, generation)],
        )

    def test_happy_path_accepts_arbitrary_positive_generation(self) -> None:
        result = probe.exercise_registered_plans(
            FakeBackend(initial_generation=0x1234), blocks=1
        )
        self.assertEqual(result["generation_start"], 0x1234)
        self.assertEqual(result["generation_end"], 0x1234 + 5)

    def test_all_negative_cases_are_isolated_and_fail_closed(self) -> None:
        for case in probe.NEGATIVE_CASES:
            with self.subTest(case=case):
                result = probe.exercise_negative_case(FakeBackend(), case=case)
                self.assertEqual(result["status"], "pass")
                self.assertEqual(result["negative_case"], case)
                self.assertTrue(result["zero_slots_after_close"])
                self.assertTrue(result["rejection"]["rejected"])
                delta = result["rejection_profile_delta"]
                if case in ("allocation_invalidation", "backing_invalidation"):
                    self.assertEqual(delta["slot_direct_ioctl_calls"], 1)
                    self.assertEqual(delta["slot_direct_errors"], 1)
                    self.assertEqual(delta["slot_direct_copy_pushes"], 0)
                    self.assertEqual(delta["slot_direct_pte_map_ranges"], 0)
                    self.assertEqual(
                        delta["slot_direct_transition_fast_index_calls"],
                        int(case == "backing_invalidation"),
                    )
                else:
                    self.assertTrue(
                        all(
                            delta[name] == 0
                            for name in probe.CORE_ACTIVITY_COUNTERS
                        )
                    )

    def test_aba_requires_monotonic_new_session_generation(self) -> None:
        result = probe.exercise_negative_case(
            FakeBackend(initial_generation=91), case="generation_aba"
        )
        self.assertGreater(result["new_generation"], result["old_generation"])
        self.assertEqual(result["rejection"]["generation"], 91)
        self.assertEqual(result["rejection"]["new_generation"], 92)
        self.assertFalse(result["rejection"]["poisoned"])

    def test_allocation_identity_preflight_preserves_active_session(self) -> None:
        result = probe.exercise_negative_case(
            FakeBackend(), case="allocation_invalidation"
        )
        self.assertTrue(result["allocation_recreated"])
        self.assertTrue(result["same_va_reused"])
        self.assertFalse(result["rejection"]["poisoned"])
        self.assertEqual(result["rejection"]["generation"], 37)
        self.assertEqual(result["rejection"]["new_generation"], 37)
        self.assertEqual(
            result["post_rejection_state"],
            {
                "allocated_slots": probe.SLICES_PER_BLOCK,
                "plan_count": 2,
                "tracked_generation": 37,
                "session_open": True,
            },
        )
        delta = result["rejection_profile_delta"]
        self.assertEqual(delta["slot_direct_copy_pushes"], 0)
        self.assertEqual(delta["slot_direct_pte_map_ranges"], 0)
        self.assertEqual(delta["slot_direct_pte_write_entries"], 0)

    def test_cached_backing_invalidation_is_fail_closed(self) -> None:
        result = probe.exercise_negative_case(
            FakeBackend(), case="backing_invalidation"
        )
        self.assertEqual(
            result["post_release_state"],
            {
                "allocated_slots": 0,
                "plan_count": 2,
                "tracked_generation": 37,
                "session_open": True,
            },
        )
        self.assertEqual(
            result["post_reallocate_state"],
            {
                "allocated_slots": probe.SLICES_PER_BLOCK,
                "plan_count": 2,
                "tracked_generation": 37,
                "session_open": True,
            },
        )
        self.assertEqual(
            result["post_rejection_state"], result["post_reallocate_state"]
        )
        self.assertFalse(result["rejection"]["poisoned"])
        self.assertEqual(result["rejection"]["new_generation"], 37)
        delta = result["rejection_profile_delta"]
        self.assertEqual(delta["slot_direct_transition_fast_index_calls"], 1)
        self.assertEqual(delta["slot_direct_copy_pushes"], 0)
        self.assertEqual(delta["slot_direct_pte_map_ranges"], 0)
        self.assertEqual(delta["slot_direct_pte_write_entries"], 0)
        self.assertEqual(result["post_close_state"]["allocated_slots"], 0)

    def test_legacy_raw_rejection_counts_both_control_paths(self) -> None:
        result = probe.exercise_negative_case(FakeBackend(), case="legacy_mode20")
        delta = result["rejection_profile_delta"]
        self.assertEqual(delta["slot_legacy_ioctl_calls"], 1)
        self.assertEqual(delta["slot_legacy_transition_calls"], 1)
        self.assertEqual(delta["slot_plan_execute_calls"], 0)

    def test_stale_generation_activity_is_rejected(self) -> None:
        with self.assertRaisesRegex(probe.ProbeError, "stale generation"):
            probe.exercise_registered_plans(StaleSideEffectBackend(), blocks=1)

    def test_stale_generation_must_reach_kernel_execute_counter(self) -> None:
        with self.assertRaisesRegex(probe.ProbeError, "slot_plan_execute_calls"):
            probe.exercise_registered_plans(
                MissingStaleExecuteCounterBackend(), blocks=1
            )

    def test_va_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(probe.ProbeError, "managed VA changed"):
            probe.exercise_registered_plans(VaMutationBackend(), blocks=1)

    def test_post_close_slot_leak_is_rejected(self) -> None:
        with self.assertRaisesRegex(probe.ProbeError, "post-close allocated_slots"):
            probe.exercise_registered_plans(CloseLeakBackend(), blocks=1)

    def test_legacy_transition_ioctl_is_rejected(self) -> None:
        with self.assertRaisesRegex(probe.ProbeError, "slot_legacy_transition_calls"):
            probe.exercise_registered_plans(LegacyTransitionLeakBackend(), blocks=1)


class KernelBackingGenerationContractTests(unittest.TestCase):
    def test_every_slot_mem_assignment_updates_or_preserves_generation(self) -> None:
        source_path = (
            probe.REPO_ROOT
            / "kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c"
        )
        source = source_path.read_text(encoding="utf-8")
        assignments = list(
            re.finditer(
                r"(?m)^\s*slot->slot_mem\s*=\s*([^;]+);",
                source,
            )
        )
        self.assertEqual(
            [match.group(1).strip() for match in assignments],
            ["NULL", "slot_mem", "new_mem", "new_mem"],
            "audit every direct slot_mem replacement/retention path",
        )

        for match in assignments:
            rhs = match.group(1).strip()
            context = source[match.start() : match.start() + 420]
            if rhs == "NULL":
                self.assertIn("slot->backing_generation = 0;", context)
            elif rhs == "slot_mem":
                self.assertIn(
                    "slot->backing_generation = backing_generation;", context
                )
            else:
                self.assertIn(
                    "slot->backing_generation = "
                    "++g_uvm_moe_slot_pool_backing_generation;",
                    context,
                )
                self.assertIn("if (slot->backing_generation == 0)", context)

        self.assertNotRegex(
            source,
            r"g_uvm_moe_slot_pool_backing_generation\s*=\s*0",
            "module-lifetime backing generation must not reset with the pool",
        )
        self.assertIn(
            "slot->backing_generation != cached->backing_generation",
            source,
        )
        self.assertIn(
            "cached->backing_generation = slot->backing_generation;",
            source,
        )


if __name__ == "__main__":
    unittest.main()
