#!/usr/bin/env python3
"""CPU-only contract test for policy state across mode-19 slot reuse.

All driver ioctls are mocked.  The test deliberately keeps two 512 KiB
regions in the same 2 MiB VA block so it covers the layer-batch mode-10
contract for disjoint slices and the hard lifecycle case: release one slot
while its neighbour remains mapped, then allocate and restore the released
region again in the same process.
"""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = REPO_ROOT / "workloads" / "vllm" / "vllm"
sys.path.insert(0, str(VLLM_ROOT))

from vllm.device_allocator import uvm as uvm_module  # noqa: E402
from vllm.device_allocator import uvm_pool_residency as residency_module  # noqa: E402
from vllm.device_allocator.uvm_region import (  # noqa: E402
    ExplicitRegionRegistry,
    RegionState,
)


MIB = 1024 * 1024
VA_BLOCK_BYTES = 2 * MIB
REGION_BYTES = 512 * 1024
UUID_BYTES = bytes(range(16))


def _gpu_uuid() -> uvm_module._PhaseFUuid:
    return uvm_module._PhaseFUuid(
        (ctypes.c_uint8 * len(UUID_BYTES)).from_buffer_copy(UUID_BYTES)
    )


class FakeSlotIoctl:
    """Complete mode 8/10/19 calls while preserving their public ABI."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.slot_bytes: dict[tuple[int, int], int] = {}

    def __call__(self, fd: int, request: int, params: object) -> int:
        mode = int(params.mode)
        entries: list[dict[str, int]] = []
        requested_bytes = 0
        accepted_bytes = 0
        for index in range(int(params.count)):
            entry = params.entries[index]
            record = {
                "va_base": int(entry.va_base),
                "bytes": int(entry.bytes),
                "layer_id": int(entry.layer_id),
                "expert_id": int(entry.expert_id),
                "slot_id": int(entry.slot_id),
                "slot_offset": int(entry.slot_offset),
                "flags": int(entry.flags),
            }
            entries.append(record)
            requested_bytes += record["bytes"]
            slot_key = (record["layer_id"], record["slot_id"])
            if mode == 8:
                self.slot_bytes[slot_key] = record["bytes"]
                accepted_bytes += record["bytes"]
            elif mode == 10:
                accepted_bytes += record["bytes"]
            elif mode == 19:
                accepted_bytes += self.slot_bytes.pop(slot_key)
            else:
                raise AssertionError(f"unexpected mocked mode: {mode}")

        self.calls.append(
            {
                "fd": int(fd),
                "request": int(request),
                "mode": mode,
                "entries": entries,
            }
        )
        params.success_count = params.count
        params.requested_bytes = requested_bytes
        params.accepted_bytes = accepted_bytes
        if mode == 10:
            params.remap_count = params.count
            params.tlb_invalidate_count = params.count
            params.reserved = params.count
        elif mode == 19:
            params.remap_count = params.count
            params.tlb_invalidate_count = params.count
        return 0


class SlotPolicyLifecycleTest(unittest.TestCase):
    def test_partial_release_clears_state_and_same_process_restore_reuses_block(
        self,
    ) -> None:
        release_slots = getattr(
            residency_module,
            "release_uvm_regions_from_slots",
            None,
        )
        self.assertTrue(
            callable(release_slots),
            "missing policy mode-19 API: release_uvm_regions_from_slots",
        )

        registry = ExplicitRegionRegistry()
        block_base = 0x40000000
        registry.register_region(
            "left",
            block_base,
            REGION_BYTES,
            RegionState.CPU,
        )
        registry.register_region(
            "right",
            block_base + REGION_BYTES,
            REGION_BYTES,
            RegionState.CPU,
        )
        self.assertEqual(block_base // VA_BLOCK_BYTES, (block_base + REGION_BYTES) // VA_BLOCK_BYTES)

        fake_ioctl = FakeSlotIoctl()
        closed_fds: list[int] = []
        bindings: dict[tuple[int, int, str], int] = {}
        owners: dict[tuple[int, int], tuple[int, int, str]] = {}
        preallocated: set[tuple[int, int, str]] = set()
        ready: set[tuple[int, int, str]] = set()
        allocation_bytes: dict[tuple[int, int, str], int] = {}
        group_last_used: dict[tuple[int, int], int] = {}
        protected_groups: dict[tuple[int, int], int] = {}

        with (
            patch.multiple(
                residency_module,
                _EXPLICIT_REGION_REGISTRY=registry,
                _SLOT_DIRECT_BINDINGS=bindings,
                _SLOT_DIRECT_SLOT_OWNERS=owners,
                _SLOT_DIRECT_PREALLOCATED=preallocated,
                _SLOT_DIRECT_READY=ready,
                _SLOT_DIRECT_ALLOCATION_BYTES=allocation_bytes,
                _SLOT_DIRECT_GROUP_LAST_USED=group_last_used,
                _SLOT_DIRECT_PROTECTED_GROUPS=protected_groups,
                _SLOT_DIRECT_USE_EPOCH=0,
                _SLOT_DIRECT_ACTIVE_DEVICE=None,
                _SLOT_DIRECT_OWNER_THREAD=None,
                _SLOT_DIRECT_POISON_REASON=None,
                _SLOT_DIRECT_CACHE_LIMITS=(4 * MIB, 16),
            ),
            patch.multiple(
                uvm_module,
                _uvm_slot_direct_sessions={},
                _uvm_slot_direct_owners={},
                _uvm_slot_direct_close_registered=True,
                _phase_f_gpu_uuid=lambda _device: _gpu_uuid(),
                _iter_existing_uvm_fd_sources=lambda: [5],
                _initialize_duplicated_uvm_fd=lambda fd: fd + 100,
                _libc_ioctl=fake_ioctl,
            ),
            patch.object(
                uvm_module,
                "_open_initialized_uvm_fd",
                side_effect=AssertionError("new UVM fd is forbidden"),
            ),
            patch.object(uvm_module.os, "close", lambda fd: closed_fds.append(fd)),
        ):
            first = residency_module.restore_uvm_regions_to_slots(
                ["left", "right"],
                layer_id=7,
                expert_id=11,
                device=0,
            )
            self.assertTrue(first["ok"], first)
            self.assertEqual(first["reason"], "ok")
            first_slot_ids = dict(first["slot_ids"])
            expected_keys = {(7, 11, "left"), (7, 11, "right")}
            self.assertEqual(set(bindings), expected_keys)
            self.assertEqual(preallocated, expected_keys)
            self.assertEqual(ready, expected_keys)
            self.assertEqual(set(allocation_bytes), expected_keys)

            # The kernel layer-batch path locks each unique VA block once and
            # accepts disjoint slices owned by different slots in one mode-10
            # transaction.
            first_mode10 = [call for call in fake_ioctl.calls if call["mode"] == 10]
            self.assertEqual(len(first_mode10), 1)
            self.assertEqual(len(first_mode10[0]["entries"]), 2)

            release_left = release_slots(
                ["left"],
                layer_id=7,
                expert_id=11,
                device=0,
            )
            self.assertTrue(release_left["ok"], release_left)
            self.assertNotIn((7, 11, "left"), bindings)
            self.assertNotIn((7, 11, "left"), preallocated)
            self.assertNotIn((7, 11, "left"), ready)
            self.assertNotIn((7, 11, "left"), allocation_bytes)
            self.assertIn((7, 11, "right"), bindings)
            self.assertIn((7, 11, "right"), preallocated)
            self.assertIn((7, 11, "right"), ready)
            self.assertIn((7, 11, "right"), allocation_bytes)

            calls_before_reuse = len(fake_ioctl.calls)
            reused = residency_module.restore_uvm_regions_to_slots(
                ["left"],
                layer_id=7,
                expert_id=11,
                device=0,
            )
            self.assertTrue(reused["ok"], reused)
            self.assertEqual(reused["reason"], "ok")
            self.assertNotEqual(reused["reason"], "already_slot_ready")
            self.assertEqual(reused["slot_ids"], {"left": first_slot_ids["left"]})
            self.assertEqual(
                [call["mode"] for call in fake_ioctl.calls[calls_before_reuse:]],
                [8, 10],
            )
            self.assertEqual(preallocated, expected_keys)
            self.assertEqual(ready, expected_keys)
            self.assertEqual(set(allocation_bytes), expected_keys)

            release_all = release_slots(
                ["left", "right"],
                layer_id=7,
                expert_id=11,
                device=0,
            )
            self.assertTrue(release_all["ok"], release_all)
            self.assertEqual(bindings, {})
            self.assertEqual(owners, {})
            self.assertEqual(preallocated, set())
            self.assertEqual(ready, set())
            self.assertEqual(allocation_bytes, {})
            self.assertEqual(group_last_used, {})
            self.assertEqual(protected_groups, {})
            self.assertEqual(uvm_module._uvm_slot_direct_owners.get(0, {}), {})

            close = uvm_module.close_uvm_slot_direct_session(device=0)
            self.assertTrue(close["ok"], close)
            self.assertTrue(close["closed"], close)
            self.assertEqual(closed_fds, [105])


if __name__ == "__main__":
    unittest.main(verbosity=2)
