from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_probe.py")
SPEC = importlib.util.spec_from_file_location("fragmented_plan_probe_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)

REPO_ROOT = Path(__file__).resolve().parents[2]


class LayoutTest(unittest.TestCase):
    def test_packed_geometry(self) -> None:
        pointers = tuple(
            0x100000000 + index * probe.base.BLOCK_BYTES
            for index in range(probe.PACKED_ALLOCATION_COUNT)
        )
        entries = probe.build_entries(
            pointers,
            layout="packed",
            logical_layer=0,
            bank_id=0,
            reuse_required=False,
        )
        metrics = probe.layout_metrics(entries)
        probe.require_layout_contract("packed", metrics)
        self.assertEqual(metrics["complete_2m_blocks"], 288)
        self.assertEqual(metrics["partial_2m_blocks"], 0)

    def test_fully_fragmented_geometry(self) -> None:
        stride = 2 * probe.base.BLOCK_BYTES
        pointers = tuple(
            0x400000000 + index * stride for index in range(probe.ENTRY_COUNT)
        )
        entries = probe.build_entries(
            pointers,
            layout="fragmented",
            logical_layer=1,
            bank_id=1,
            reuse_required=True,
        )
        metrics = probe.layout_metrics(entries)
        probe.require_layout_contract("fragmented", metrics)
        self.assertEqual(metrics["va_block_count"], 1152)
        self.assertEqual(metrics["complete_2m_blocks"], 0)
        self.assertEqual(metrics["partial_2m_blocks"], 1152)
        self.assertEqual(metrics["adjacent_range_pairs"], 0)

    def test_gapped_complete_block_geometry(self) -> None:
        pointers = tuple(
            0x300000000 + index * probe.GAPPED_ALLOCATION_BYTES
            for index in range(probe.PACKED_ALLOCATION_COUNT)
        )
        entries = probe.build_entries(
            pointers,
            layout="gapped",
            logical_layer=0,
            bank_id=0,
            reuse_required=False,
        )
        metrics = probe.layout_metrics(entries)
        probe.require_layout_contract("gapped", metrics)
        self.assertEqual(metrics["complete_2m_blocks"], 288)
        self.assertEqual(metrics["partial_2m_blocks"], 0)
        self.assertEqual(metrics["adjacent_range_pairs"], 288 * 3)
        self.assertEqual(metrics["nonzero_gap_pairs"], 287)
        self.assertEqual(
            metrics["minimum_positive_gap_bytes"], probe.base.BLOCK_BYTES
        )

    def test_fragmented_contract_rejects_two_entries_in_one_block(self) -> None:
        pointers = tuple(
            0x800000000 + index * 2 * probe.base.BLOCK_BYTES
            for index in range(probe.ENTRY_COUNT)
        )
        pointers = (pointers[0], pointers[0] + probe.base.SLICE_BYTES) + pointers[2:]
        entries = probe.build_entries(
            pointers,
            layout="fragmented",
            logical_layer=0,
            bank_id=0,
            reuse_required=False,
        )
        with self.assertRaises(probe.ProbeError):
            probe.require_layout_contract("fragmented", probe.layout_metrics(entries))

    def test_payload_shape(self) -> None:
        self.assertEqual(probe.ENTRY_COUNT, 128 * (6 + 3))
        self.assertEqual(probe.LOGICAL_BYTES, 576 * 1024 * 1024)


class QwenPackedSparseTest(unittest.TestCase):
    def _entries(
        self,
        expert_ids: tuple[int, ...],
        *,
        w13_offset: int = 0,
        w2_offset: int = 0,
    ) -> tuple[tuple[tuple[int, ...], ...], dict[str, object]]:
        w13_base = 0x1000000000 + w13_offset
        w2_base = 0x2000000000 + w2_offset
        pointers = probe.qwen_packed_region_pointers(
            w13_base, w2_base, expert_ids
        )
        entries = probe.build_entries(
            pointers,
            layout="qwen-packed-sparse",
            logical_layer=2,
            bank_id=1,
            reuse_required=True,
            expert_ids=expert_ids,
        )
        metrics = probe.layout_metrics(entries)
        probe.require_layout_contract(
            "qwen-packed-sparse",
            metrics,
            expected_entry_count=len(expert_ids) * probe.ENVELOPES_PER_EXPERT,
            expected_logical_bytes=(
                len(expert_ids)
                * probe.ENVELOPES_PER_EXPERT
                * probe.base.SLICE_BYTES
            ),
        )
        return entries, metrics

    def test_real_fp8_shapes_strides_and_bytes(self) -> None:
        self.assertEqual(probe.QWEN_W13_SHAPE, (128, 1536, 2048))
        self.assertEqual(probe.QWEN_W2_SHAPE, (128, 2048, 768))
        self.assertEqual(probe.QWEN_W13_EXPERT_BYTES, 3 * 1024 * 1024)
        self.assertEqual(probe.QWEN_W2_EXPERT_BYTES, 1536 * 1024)
        self.assertEqual(probe.QWEN_W13_TENSOR_BYTES, 384 * 1024 * 1024)
        self.assertEqual(probe.QWEN_W2_TENSOR_BYTES, 192 * 1024 * 1024)
        self.assertEqual(probe.QWEN_MAIN_TENSOR_BYTES, 576 * 1024 * 1024)

        pointers = probe.qwen_packed_region_pointers(
            0x1000000000, 0x2000000000, (0, 1)
        )
        self.assertEqual(
            pointers[9] - pointers[0], probe.QWEN_W13_EXPERT_BYTES
        )
        self.assertEqual(
            pointers[15] - pointers[6], probe.QWEN_W2_EXPERT_BYTES
        )

    def test_arbitrary_consecutive_experts_form_full_and_partial_blocks(self) -> None:
        expert_ids = (0, 1)
        entries, metrics = self._entries(expert_ids)
        self.assertEqual(len(entries), 18)
        self.assertEqual(metrics["va_block_count"], 5)
        self.assertEqual(metrics["complete_2m_blocks"], 4)
        self.assertEqual(metrics["partial_2m_blocks"], 1)
        self.assertEqual(metrics["multi_entry_2m_blocks"], 5)
        self.assertEqual(metrics["max_entry_slices_per_2m_block"], 4)
        self.assertTrue(metrics["mixed_full_partial_2m_layout"])
        self.assertEqual(
            metrics["cold_64k_leaf_mask_histogram"],
            {"0xffffffff": 4, "0x0000ffff": 1},
        )
        owners = tuple(int(entry[3]) for entry in entries)
        owner_base = 0x10000 * 3
        self.assertEqual(owners[:9], (owner_base,) * 9)
        self.assertEqual(owners[9:], (owner_base + 1,) * 9)

    def test_noncontiguous_set_and_non_2m_tensor_bases(self) -> None:
        expert_ids = (0, 7, 31, 64, 127)
        _entries, metrics = self._entries(
            expert_ids,
            w13_offset=probe.base.SLICE_BYTES,
            w2_offset=3 * probe.base.SLICE_BYTES,
        )
        self.assertEqual(metrics["entry_count"], 45)
        self.assertEqual(metrics["va_block_count"], 20)
        self.assertEqual(metrics["complete_2m_blocks"], 3)
        self.assertEqual(metrics["partial_2m_blocks"], 17)
        self.assertEqual(metrics["multi_entry_2m_blocks"], 12)
        self.assertEqual(metrics["partial_preserve_64k_leaf_entries"], 280)
        self.assertTrue(metrics["mixed_full_partial_2m_layout"])

    def test_full_expert_set_recovers_two_dense_tensors(self) -> None:
        _entries, metrics = self._entries(tuple(range(probe.TOTAL_EXPERTS)))
        self.assertEqual(metrics["entry_count"], 1152)
        self.assertEqual(metrics["va_block_count"], 288)
        self.assertEqual(metrics["complete_2m_blocks"], 288)
        self.assertEqual(metrics["partial_2m_blocks"], 0)
        self.assertEqual(metrics["max_entry_slices_per_2m_block"], 4)

    def test_tensor_base_recovery_and_summary(self) -> None:
        expert_ids = (3, 4, 19)
        w13_base = 0x1000000000 + probe.base.SLICE_BYTES
        w2_base = 0x2000000000 + 2 * probe.base.SLICE_BYTES
        pointers = probe.qwen_packed_region_pointers(
            w13_base, w2_base, expert_ids
        )
        self.assertEqual(
            probe.qwen_tensor_bases_from_regions(pointers, expert_ids),
            (w13_base, w2_base),
        )
        summary = probe.qwen_tensor_layout_summary(pointers, expert_ids)
        self.assertEqual(summary["w13"]["shape"], [128, 1536, 2048])
        self.assertEqual(
            summary["w13"]["stride"], [3145728, 2048, 1]
        )
        self.assertEqual(summary["w2"]["stride"], [1572864, 768, 1])
        self.assertEqual(summary["w13"]["base_offset_in_2m"], 512 * 1024)
        self.assertEqual(summary["w2"]["base_offset_in_2m"], 1024 * 1024)

    def test_registration_residency_partition_is_exact(self) -> None:
        expert_ids = (0, 7, 31, 64, 127)
        cold = probe.qwen_packed_region_pointers(
            0x1000000000 + probe.base.SLICE_BYTES,
            0x2000000000 + 3 * probe.base.SLICE_BYTES,
            expert_ids,
        )
        full, observed_cold, hot = probe.qwen_registration_residency_partition(
            cold, expert_ids
        )
        self.assertEqual(observed_cold, cold)
        self.assertEqual(len(full), 128 * 9)
        self.assertEqual(len(cold), 5 * 9)
        self.assertEqual(len(hot), 123 * 9)
        self.assertEqual(set(full), set(cold) | set(hot))
        self.assertFalse(set(cold) & set(hot))
        self.assertEqual(len(set(full)), len(full))

    def test_registration_residency_setup_orders_touch_before_cold_cpu(self) -> None:
        expert_ids = (0, 7)
        cold = probe.qwen_packed_region_pointers(
            0x1000000000, 0x2000000000, expert_ids
        )
        full, _cold, hot = probe.qwen_registration_residency_partition(
            cold, expert_ids
        )
        operations: list[object] = []

        class FakeLibrary:
            def __init__(self) -> None:
                self.cpu_prefetched = False
                self.prefetched: tuple[int, ...] = ()
                self.batch_calls = 0
                self.runtime_calls = 0

            def uvm512_prefetch_counter_reset(self) -> None:
                operations.append("counter_reset")
                self.batch_calls = 0
                self.runtime_calls = 0

            def uvm512_prefetch_batch(
                self,
                pointer_array: object,
                size_array: object,
                count: int,
                device: int,
                stream: int,
            ) -> int:
                selected = tuple(int(pointer_array[index]) for index in range(count))
                sizes = tuple(int(size_array[index]) for index in range(count))
                operations.append(("prefetch_cpu", selected, sizes, device, stream))
                self.prefetched = selected
                self.cpu_prefetched = True
                self.batch_calls += 1
                self.runtime_calls += count
                return count

            def uvm512_stream_synchronize(self, stream: int) -> int:
                operations.append(("synchronize", stream))
                return 0

            def uvm512_prefetch_one_outer_calls(self) -> int:
                return 0

            def uvm512_prefetch_batch_outer_calls(self) -> int:
                return self.batch_calls

            def uvm512_runtime_prefetch_calls(self) -> int:
                return self.runtime_calls

        library = FakeLibrary()
        cold_set = set(cold)

        class FakeUvm:
            @staticmethod
            def driver_batch_query_ranges_residency_bytes(
                ranges: object,
                device: int,
                *,
                wait_for_trackers: bool,
            ) -> dict[str, int | bool]:
                selected = tuple(int(item[0]) for item in ranges)
                operations.append(("query", selected, device, wait_for_trackers))
                selected_bytes = len(selected) * probe.base.SLICE_BYTES
                cpu_bytes = 0
                gpu_bytes = selected_bytes
                if library.cpu_prefetched:
                    cpu_count = sum(pointer in cold_set for pointer in selected)
                    cpu_bytes = cpu_count * probe.base.SLICE_BYTES
                    gpu_bytes = selected_bytes - cpu_bytes
                return {
                    "ok": True,
                    "status": len(selected),
                    "queried_bytes": selected_bytes,
                    "gpu_bytes": gpu_bytes,
                    "cpu_bytes": cpu_bytes,
                    "any_bytes": selected_bytes,
                    "unknown_bytes": 0,
                }

        class FakeBackend:
            def __init__(self) -> None:
                self.library = library
                self.uvm = FakeUvm()
                self.device = 3
                self.stream = 91

            def checksum(self, dataset: int) -> tuple[int, int]:
                operations.append(("checksum", dataset))
                return 0x12345678, 0x12345678

            @staticmethod
            def _check_cuda(status: int, operation: str) -> None:
                if status != 0:
                    raise AssertionError(operation)

        shape = probe.WorkloadShape(
            layout="qwen-packed-sparse",
            expert_ids=expert_ids,
            entry_count=len(cold),
            logical_bytes=len(cold) * probe.base.SLICE_BYTES,
        )
        result = probe.prepare_qwen_registration_residency(
            FakeBackend(), {0: 17}, {0: cold}, {0: shape}
        )

        self.assertEqual(operations[0], ("checksum", 17))
        self.assertEqual(operations[1][0], "query")
        self.assertEqual(tuple(operations[1][1]), full)
        self.assertEqual(operations[2], "counter_reset")
        self.assertEqual(operations[3][0], "prefetch_cpu")
        self.assertEqual(tuple(operations[3][1]), cold)
        self.assertEqual(set(operations[3][1]), cold_set)
        self.assertEqual(operations[3][3], probe.CUDA_CPU_DEVICE_ID)
        self.assertEqual(operations[4], ("synchronize", 91))
        self.assertEqual(tuple(operations[5][1]), cold)
        self.assertEqual(tuple(operations[6][1]), hot)
        self.assertEqual(library.prefetched, cold)
        self.assertEqual(
            result["prefetch_counters"],
            {
                "one_outer_calls": 0,
                "batch_outer_calls": 1,
                "runtime_prefetch_calls": len(cold),
            },
        )
        layer = result["layers"]["0"]
        self.assertEqual(layer["post_gpu_touch_residency"]["gpu_bytes"], 576 * 1024 * 1024)
        self.assertEqual(layer["final_cold_residency"]["cpu_bytes"], len(cold) * probe.base.SLICE_BYTES)
        self.assertEqual(layer["final_cold_residency"]["gpu_bytes"], 0)
        self.assertEqual(layer["final_hot_residency"]["gpu_bytes"], len(hot) * probe.base.SLICE_BYTES)
        self.assertTrue(
            result["contract"]["full_tensor_gpu_touch_before_cold_prefetch"]
        )
        self.assertTrue(
            result["contract"]["only_selected_cold_regions_prefetched_to_cpu"]
        )

    def test_residency_query_rejects_partial_positive_status(self) -> None:
        class FakeUvm:
            @staticmethod
            def driver_batch_query_ranges_residency_bytes(
                ranges: object,
                device: int,
                *,
                wait_for_trackers: bool,
            ) -> dict[str, int | bool]:
                count = len(tuple(ranges))
                requested = count * probe.base.SLICE_BYTES
                return {
                    "ok": True,
                    "status": count - 1,
                    "queried_bytes": requested,
                    "gpu_bytes": requested,
                    "cpu_bytes": 0,
                    "any_bytes": requested,
                    "unknown_bytes": 0,
                }

        class FakeBackend:
            uvm = FakeUvm()
            device = 0

        with self.assertRaises(probe.ProbeError):
            probe.query_qwen_region_residency(
                FakeBackend(), (0x1000000000, 0x1000800000), label="partial"
            )

    def test_workload_accepts_arbitrary_consecutive_cold_set(self) -> None:
        shapes = probe.workload_shapes_by_layer(
            "qwen-packed-sparse",
            cold_expert_count=3,
            cold_expert_ids="5-7",
            cold_expert_ids_by_layer="",
        )
        self.assertEqual({shape.expert_ids for shape in shapes.values()}, {(5, 6, 7)})
        self.assertEqual({shape.entry_count for shape in shapes.values()}, {27})
        args = probe.parse_args(
            [
                "--layout",
                "qwen-packed-sparse",
                "--pte-mode",
                "partial64k",
                "--cold-expert-ids",
                "5-7",
                "--qwen-base-offset-cycle",
            ]
        )
        self.assertTrue(args.qwen_base_offset_cycle)
        command = probe.worker_command(args, Path("/tmp/qwen-worker.json"))
        self.assertIn("--qwen-base-offset-cycle", command)

    def test_qwen_dataset_abi_and_constructor_arguments(self) -> None:
        calls: list[tuple[object, ...]] = []

        class FakeOperation:
            argtypes: object = None
            restype: object = None

            def __call__(self, *args: object) -> int:
                calls.append(args)
                return 23

        class FakeLibrary:
            uvm512_dataset_create_qwen_packed_sparse = FakeOperation()

            def uvm512_dataset_reseed(self, handle: int, seed: int) -> int:
                calls.append(("reseed", handle, seed))
                return 0

        class FakeBackend:
            def __init__(self) -> None:
                self.library = FakeLibrary()
                self.datasets: list[int] = []

            def _check_cuda(self, status: int, operation: str) -> None:
                self.check = (status, operation)

            def dataset_pointers(self, handle: int, count: int) -> tuple[int, ...]:
                self.request = (handle, count)
                return probe.qwen_packed_region_pointers(
                    0x1000000000 + probe.base.SLICE_BYTES,
                    0x2000000000 + 3 * probe.base.SLICE_BYTES,
                    (0, 7),
                )

        backend = FakeBackend()
        handle, pointers = probe.create_dataset(
            backend,
            layout="qwen-packed-sparse",
            layer_id=1,
            seed=91,
            guard_bytes=0,
            entry_count=18,
            expert_ids=(0, 7),
            qwen_base_offsets=(
                probe.base.SLICE_BYTES,
                3 * probe.base.SLICE_BYTES,
            ),
        )
        self.assertEqual(handle, 23)
        self.assertEqual(len(pointers), 18)
        self.assertEqual(backend.datasets, [23])
        self.assertEqual(calls[0][1:], (2, 128, 524288, 6, 3, 524288, 1572864))
        self.assertEqual(calls[1], ("reseed", 23, 91))
        operation = backend.library.uvm512_dataset_create_qwen_packed_sparse
        self.assertEqual(len(operation.argtypes), 8)
        self.assertIs(operation.restype, probe.base.ctypes.c_void_p)

    def test_mixed_plan_shadow_is_exactly_full_blocks(self) -> None:
        expert_ids = (0, 1)
        _entries, metrics = self._entries(expert_ids)
        contract = probe.pte_contract("partial64k", metrics)
        entry_count = 18
        logical_bytes = entry_count * probe.base.SLICE_BYTES
        base_delta = {
            "slot_plan_execute_calls": 1,
            "slot_direct_ioctl_calls": 1,
            "slot_direct_successes": 1,
            "slot_direct_errors": 0,
            "slot_direct_fallbacks": 0,
            "slot_direct_requested_ranges": entry_count,
            "slot_direct_completed_ranges": entry_count,
            "slot_direct_requested_bytes": logical_bytes,
            "slot_direct_completed_bytes": logical_bytes,
            "slot_direct_dual_bank_copy_first_calls": 1,
            "slot_direct_pte_4k_blocks": 0,
            "slot_direct_pte_64k_blocks": 5,
            "slot_direct_partial_64k_blocks": 1,
            "slot_direct_pte_write_entries": 144,
        }
        raw_result = {
            "ok": True,
            "plan_id": 10,
            "layer_id": 1,
            "generation": 7,
            "new_generation": 8,
            "mode": probe.base.MODE_TRANSITION,
            "requested_count": entry_count,
            "success_count": entry_count,
            "failure_count": 0,
            "requested_bytes": logical_bytes,
            "accepted_bytes": logical_bytes,
            "remap_count": entry_count,
            "ioctl_ret": 0,
            "rm_status": 0,
            "poisoned": False,
            "plan_execute": True,
            "copy_push_count": 1,
        }

        for shadow_requested, observed_shadow in ((False, 0), (True, 4)):
            with self.subTest(shadow_requested=shadow_requested):
                record = {
                    "_raw_result": raw_result,
                    "latency_stats_delta": {
                        **base_delta,
                        "slot_direct_shadow_pte_prefill_blocks": observed_shadow,
                    },
                }
                probe.validate_execute(
                    record,
                    plan_id=10,
                    layer_id=1,
                    generation=7,
                    initial_mode=probe.DEFAULT_PARTIAL_64K_INITIAL_MODE,
                    initial=False,
                    dual_bank_event=True,
                    entry_count=entry_count,
                    logical_bytes=logical_bytes,
                    pte_mode="partial64k",
                    expected_pte=contract,
                    shadow_requested=shadow_requested,
                )

        for wrong_shadow in (0, 5):
            record = {
                "_raw_result": raw_result,
                "latency_stats_delta": {
                    **base_delta,
                    "slot_direct_shadow_pte_prefill_blocks": wrong_shadow,
                },
            }
            with self.assertRaises(probe.ProbeError):
                probe.validate_execute(
                    record,
                    plan_id=10,
                    layer_id=1,
                    generation=7,
                    initial_mode=probe.DEFAULT_PARTIAL_64K_INITIAL_MODE,
                    initial=False,
                    dual_bank_event=True,
                    entry_count=entry_count,
                    logical_bytes=logical_bytes,
                    pte_mode="partial64k",
                    expected_pte=contract,
                    shadow_requested=True,
                )

        registration = {
            "latency_stats_delta": {
                "slot_plan_pte_prewarm_calls": 1,
                "slot_plan_pte_prewarm_successes": 1,
                "slot_plan_pte_prewarm_failures": 0,
                "slot_plan_pte_prewarm_ranges": entry_count,
                "slot_plan_pte_prewarm_blocks": 5,
                "slot_plan_pte_prewarm_allocated_blocks": 3,
                "slot_plan_pte_prewarm_reused_blocks": 2,
                "slot_plan_pte_prewarm_full_blocks": 4,
                "slot_plan_pte_prewarm_partial_blocks": 1,
                "slot_plan_pte_prewarm_bytes": logical_bytes,
            }
        }
        probe.validate_registration_prewarm(
            registration,
            pte_mode="partial64k",
            entry_count=entry_count,
            logical_bytes=logical_bytes,
            expected_pte=contract,
        )
        self.assertEqual(registration["pte_prewarm_observed"]["full_blocks"], 4)
        self.assertEqual(
            registration["pte_prewarm_observed"]["partial_blocks"], 1
        )


class KernelFaultGuardContractTest(unittest.TestCase):
    @staticmethod
    def _overlaps(base: int, size: int, other_base: int, other_size: int) -> bool:
        return base < other_base + other_size and other_base < base + size

    def test_mixed_block_hot_leaf_does_not_overlap_cold_region(self) -> None:
        block_base = 0x100000000
        cold_base = block_base
        cold_size = probe.base.SLICE_BYTES
        hot_address = block_base + 3 * probe.base.SLICE_BYTES // 2
        leaf_base = hot_address & ~(64 * 1024 - 1)

        self.assertTrue(
            self._overlaps(block_base, probe.base.BLOCK_BYTES, cold_base, cold_size)
        )
        self.assertFalse(
            self._overlaps(leaf_base, 64 * 1024, cold_base, cold_size)
        )

        cold_address = cold_base + 17
        cold_leaf_base = cold_address & ~(64 * 1024 - 1)
        self.assertTrue(
            self._overlaps(cold_leaf_base, 64 * 1024, cold_base, cold_size)
        )

    def test_gpu_fault_guards_use_one_exact_64k_leaf(self) -> None:
        cases = (
            (
                "uvm_gpu_replayable_faults.c",
                "current_entry->fault_address",
                "current_entry->va_space",
            ),
            (
                "uvm_gpu_non_replayable_faults.c",
                "fault_entry->fault_address",
                "va_space",
            ),
        )
        source_dir = (
            REPO_ROOT
            / "kernel-module"
            / "nvidia-module"
            / "kernel-open"
            / "nvidia-uvm"
        )

        for filename, address, clear_va_space in cases:
            with self.subTest(filename=filename):
                source = (source_dir / filename).read_text()
                alignment = re.compile(
                    r"fault_leaf_base\s*=\s*UVM_ALIGN_DOWN\("
                    + re.escape(address)
                    + r",\s*UVM_PAGE_SIZE_64K\)"
                )
                lookup = re.compile(
                    r"uvm_moe_slot_pool_range_has_mappings\(va_space,\s*"
                    r"fault_leaf_base,\s*UVM_PAGE_SIZE_64K\)"
                )
                clear = re.compile(
                    r"uvm_moe_slot_pool_clear_range_mappings_for_fault\("
                    + re.escape(clear_va_space)
                    + r",\s*fault_leaf_base,\s*UVM_PAGE_SIZE_64K\)"
                )

                self.assertRegex(source, alignment)
                self.assertRegex(source, lookup)
                self.assertRegex(source, clear)
                self.assertNotRegex(
                    source,
                    re.compile(
                        r"uvm_moe_slot_pool_(?:range_has_mappings|"
                        r"clear_range_mappings_for_fault)\([^;]*"
                        r"UVM_VA_BLOCK_ALIGN_DOWN",
                        re.DOTALL,
                    ),
                )

    def test_clear_helpers_invalidate_once_only_after_real_overlap(self) -> None:
        source_path = (
            REPO_ROOT
            / "kernel-module"
            / "nvidia-module"
            / "kernel-open"
            / "nvidia-uvm"
            / "uvm.c"
        )
        source = source_path.read_text()
        nonlocked_start = source.index(
            "static NV_STATUS uvm_moe_slot_pool_clear_range_mappings("
        )
        locked_start = source.index(
            "static NV_STATUS uvm_moe_slot_pool_clear_range_mappings_locked("
        )
        forget_start = source.index(
            "static NV_STATUS uvm_moe_slot_pool_forget_range_mappings_locked("
        )
        batch_migrate_start = source.index(
            "NV_STATUS uvm_moe_slot_pool_clear_ranges_mappings_for_migrate_locked("
        )
        discard_start = source.index(
            "NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_discard("
        )

        helpers = (
            source[nonlocked_start:locked_start],
            source[locked_start:forget_start],
        )
        for helper in helpers:
            with self.subTest(helper=helper.split("(", 1)[0].split()[-1]):
                self.assertIn("NvBool plan_invalidated = NV_FALSE;", helper)
                self.assertEqual(
                    helper.count("uvm_va_space_moe_slot_plan_invalidate(va_space);"),
                    1,
                )
                overlap = helper.index("uvm_moe_slot_pool_pte_mapping_overlaps")
                guard = helper.index("if (!plan_invalidated)")
                invalidate = helper.index(
                    "uvm_va_space_moe_slot_plan_invalidate(va_space);"
                )
                release = helper.index(
                    "uvm_moe_slot_pool_release_pte_mapping_locked"
                )
                self.assertLess(overlap, guard)
                self.assertLess(guard, invalidate)
                self.assertLess(invalidate, release)

        # Teardown and batched migration mutate plan lifetime/source backing
        # independently of whether a slot PTE is live, so their epoch guards
        # intentionally remain unconditional.
        forget = source[forget_start:batch_migrate_start]
        batch_migrate = source[batch_migrate_start:discard_start]
        self.assertEqual(
            forget.count("uvm_va_space_moe_slot_plan_invalidate(va_space);"), 1
        )
        self.assertEqual(
            batch_migrate.count(
                "uvm_va_space_moe_slot_plan_invalidate(va_space);"
            ),
            1,
        )

    def test_hot_fault_map_cannot_merge_over_neighboring_slot_leaf(self) -> None:
        source_path = (
            REPO_ROOT
            / "kernel-module"
            / "nvidia-module"
            / "kernel-open"
            / "nvidia-uvm"
            / "uvm_va_block.c"
        )
        source = source_path.read_text()
        map_start = source.index("static NV_STATUS block_map_gpu_to(")
        next_function = source.index(
            "static NvU64 block_moe_slot_pte_mode_page_size(", map_start
        )
        map_helper = source[map_start:next_function]

        exclude_slot_pages = map_helper.index(
            "&gpu_state->moe_slot_pte_bits);"
        )
        compute_state = map_helper.index("block_gpu_compute_new_pte_state(")
        self.assertIn("uvm_page_mask_andnot", map_helper[:exclude_slot_pages])
        self.assertLess(exclude_slot_pages, compute_state)


class SparseExpertTest(unittest.TestCase):
    def test_active_offset_cycle_and_histogram(self) -> None:
        offsets = probe.active_offsets(10, cycle=True)
        self.assertEqual(
            offsets,
            (
                0,
                512 * 1024,
                1024 * 1024,
                1536 * 1024,
                0,
                512 * 1024,
                1024 * 1024,
                1536 * 1024,
                0,
                512 * 1024,
            ),
        )
        summary = probe.active_offset_summary(probe.active_offsets(45, cycle=True))
        self.assertEqual(
            summary["histogram"],
            {
                "0": 12,
                str(512 * 1024): 11,
                str(1024 * 1024): 11,
                str(1536 * 1024): 11,
            },
        )
        self.assertEqual(summary["nonzero_offset_ranges"], 33)
        self.assertEqual(summary["two_mib_aligned_ranges"], 12)

        args = probe.parse_args(
            [
                "--layout",
                "sparse-experts",
                "--pte-mode",
                "partial64k",
                "--cold-expert-ids",
                "0,7,31,64,127",
                "--cycle-active-offsets",
            ]
        )
        self.assertTrue(args.cycle_active_offsets)
        self.assertIn(
            "--cycle-active-offsets",
            probe.worker_command(args, Path("/tmp/offset-worker.json")),
        )

    def test_cycled_partial_layout_allows_cross_allocation_adjacency(self) -> None:
        entry_count = probe.ENVELOPES_PER_EXPERT
        bases = tuple(
            0xC00000000 + index * probe.FRAGMENT_ALLOCATION_BYTES
            for index in range(entry_count)
        )
        pointers = tuple(
            base + offset
            for base, offset in zip(
                bases, probe.active_offsets(entry_count, cycle=True)
            )
        )
        entries = probe.build_entries(
            pointers,
            layout="sparse-experts",
            logical_layer=0,
            bank_id=0,
            reuse_required=False,
            # build_entries requires complete nine-envelope owner groups.
            expert_ids=(0,),
        )
        metrics = probe.layout_metrics(entries)
        self.assertGreater(metrics["adjacent_range_pairs"], 0)
        with self.assertRaises(probe.ProbeError):
            probe.require_layout_contract(
                "sparse-experts",
                metrics,
                expected_entry_count=entry_count,
                expected_logical_bytes=entry_count * probe.base.SLICE_BYTES,
            )
        probe.require_layout_contract(
            "sparse-experts",
            metrics,
            expected_entry_count=entry_count,
            expected_logical_bytes=entry_count * probe.base.SLICE_BYTES,
            allow_adjacent_partial=True,
        )

    def test_cycled_dataset_uses_offset_view_abi(self) -> None:
        calls: list[tuple[object, ...]] = []

        class FakeLibrary:
            def uvm512_dataset_create_offset_cycle(self, *args: object) -> int:
                calls.append(args)
                return 17

            def uvm512_dataset_reseed(self, handle: int, seed: int) -> int:
                calls.append(("reseed", handle, seed))
                return 0

        class FakeBackend:
            def __init__(self) -> None:
                self.library = FakeLibrary()
                self.datasets: list[int] = []

            def _check_cuda(self, status: int, operation: str) -> None:
                self.assert_status = (status, operation)

            def dataset_pointers(self, handle: int, count: int) -> tuple[int, ...]:
                self.pointer_request = (handle, count)
                return tuple(
                    0xD00000000
                    + index * probe.FRAGMENT_ALLOCATION_BYTES
                    + offset
                    for index, offset in enumerate(
                        probe.active_offsets(count, cycle=True)
                    )
                )

        backend = FakeBackend()
        handle, pointers = probe.create_dataset(
            backend,
            layout="sparse-experts",
            layer_id=2,
            seed=123,
            guard_bytes=0,
            entry_count=9,
            cycle_active_offsets=True,
        )
        self.assertEqual(handle, 17)
        self.assertEqual(len(pointers), 9)
        self.assertEqual(
            calls[0],
            (
                9,
                probe.FRAGMENT_ALLOCATION_BYTES,
                probe.base.SLICE_BYTES,
                probe.ACTIVE_OFFSET_STEP_BYTES,
                probe.ACTIVE_OFFSET_PERIOD,
                0,
            ),
        )
        self.assertEqual(calls[1], ("reseed", 17, 123))
        self.assertEqual(backend.datasets, [17])

    def test_offset_dataset_abi_uses_pointer_sized_return(self) -> None:
        class FakeOperation:
            argtypes: object = None
            restype: object = None

        class FakeLibrary:
            uvm512_dataset_create_offset_cycle = FakeOperation()

        operation = probe.bind_offset_dataset_abi(FakeLibrary())
        self.assertEqual(operation.argtypes, [probe.base.ctypes.c_size_t] * 6)
        self.assertIs(operation.restype, probe.base.ctypes.c_void_p)
        with self.assertRaises(probe.ProbeError):
            probe.bind_offset_dataset_abi(object())

    def test_guard_active_sources_restore_is_sorted_and_exact(self) -> None:
        calls: list[int] = []

        class FakeGuard:
            def __init__(self, layer_id: int, ranges: int) -> None:
                self.layer_id = layer_id
                self.active_range_count = ranges
                self.active_bytes = ranges * probe.base.SLICE_BYTES
                self.complement_prefix_bytes = 0
                self.complement_suffix_bytes = (
                    ranges * (probe.FRAGMENT_ALLOCATION_BYTES - probe.base.SLICE_BYTES)
                )

            def restore_active_cpu(self) -> None:
                calls.append(self.layer_id)

        records = probe.restore_guard_active_sources_cpu(
            {
                7: FakeGuard(7, 45),
                1: FakeGuard(1, 9),
            }
        )
        self.assertEqual(calls, [1, 7])
        self.assertEqual(
            records,
            [
                {
                    "layer_id": 1,
                    "range_count": 9,
                    "bytes": 9 * probe.base.SLICE_BYTES,
                    "complement_prefix_bytes": 0,
                    "complement_suffix_bytes": 9
                    * (probe.FRAGMENT_ALLOCATION_BYTES - probe.base.SLICE_BYTES),
                },
                {
                    "layer_id": 7,
                    "range_count": 45,
                    "bytes": 45 * probe.base.SLICE_BYTES,
                    "complement_prefix_bytes": 0,
                    "complement_suffix_bytes": 45
                    * (probe.FRAGMENT_ALLOCATION_BYTES - probe.base.SLICE_BYTES),
                },
            ],
        )

    def test_parse_expert_ids_and_ranges(self) -> None:
        self.assertEqual(
            probe.parse_expert_ids("0, 4, 8-10, 0x7f,4"),
            (0, 4, 8, 9, 10, 127),
        )

    def test_parse_expert_ids_rejects_invalid_selections(self) -> None:
        for value in ("", "9-7", "0,128"):
            with self.subTest(value=value), self.assertRaises(probe.ProbeError):
                probe.parse_expert_ids(value)

    def test_interleaved_selection_remains_sparse(self) -> None:
        self.assertEqual(probe.interleaved_expert_ids(5), (0, 2, 4, 6, 8))
        almost_all = probe.interleaved_expert_ids(127)
        self.assertEqual(len(almost_all), 127)
        self.assertTrue(
            any(
                almost_all[index] - almost_all[index - 1] > 1
                for index in range(1, len(almost_all))
            )
        )
        self.assertEqual(probe.interleaved_expert_ids(128), tuple(range(128)))

    def test_deterministic_layer_selections_differ_with_equal_geometry(self) -> None:
        for count in (5, 32, 64, 127):
            with self.subTest(count=count):
                selections = probe.deterministic_layer_expert_ids(count)
                self.assertEqual(tuple(selections), tuple(probe.base.LAYER_IDS))
                self.assertEqual({len(values) for values in selections.values()}, {count})
                self.assertEqual(len(set(selections.values())), 4)
                for values in selections.values():
                    self.assertTrue(
                        any(
                            values[index] - values[index - 1] > 1
                            for index in range(1, len(values))
                        )
                    )

        full = probe.deterministic_layer_expert_ids(128)
        self.assertEqual(set(full.values()), {tuple(range(128))})

    def test_parse_per_layer_expert_ids_and_reject_geometry_mismatch(self) -> None:
        selections = probe.parse_layer_expert_ids(
            "L0=0,7,31;L1=1,8,32;L2=2,9,33;L3=3,10,34"
        )
        self.assertEqual(selections[0], (0, 7, 31))
        self.assertEqual(selections[3], (3, 10, 34))

        for value in (
            "L0=0,7;L1=1,8;L2=2,9",
            "L0=0,7;L1=1,8;L2=2,9;L3=3,10,34",
            "L0=0,7;L0=1,8;L2=2,9;L3=3,10",
        ):
            with self.subTest(value=value), self.assertRaises(probe.ProbeError):
                probe.parse_layer_expert_ids(value)

    def test_workload_shapes_use_per_layer_ids_and_default_to_different_sets(self) -> None:
        explicit = probe.workload_shapes_by_layer(
            "sparse-experts",
            cold_expert_count=16,
            cold_expert_ids="",
            cold_expert_ids_by_layer=(
                "L0=0,7,31,64,127;L1=1,8,32,65,126;"
                "L2=2,9,33,66,125;L3=3,10,34,67,124"
            ),
        )
        self.assertEqual({shape.entry_count for shape in explicit.values()}, {45})
        self.assertEqual(len({shape.expert_ids for shape in explicit.values()}), 4)

        generated = probe.workload_shapes_by_layer(
            "sparse-experts",
            cold_expert_count=32,
            cold_expert_ids="",
            cold_expert_ids_by_layer="",
        )
        self.assertEqual({shape.entry_count for shape in generated.values()}, {288})
        self.assertEqual(len({shape.expert_ids for shape in generated.values()}), 4)

        with self.assertRaises(probe.ProbeError):
            probe.workload_shapes_by_layer(
                "sparse-experts",
                cold_expert_count=5,
                cold_expert_ids="0,2,4,6,8",
                cold_expert_ids_by_layer=(
                    "L0=0,7;L1=1,8;L2=2,9;L3=3,10"
                ),
            )

    def test_sparse_shape_and_qwen_owner_groups(self) -> None:
        expert_ids = (0, 7, 31, 64, 127)
        shape = probe.workload_shape(
            "sparse-experts",
            cold_expert_count=16,
            cold_expert_ids=",".join(str(value) for value in expert_ids),
        )
        self.assertEqual(shape.entry_count, 45)
        self.assertEqual(shape.logical_bytes, 45 * probe.base.SLICE_BYTES)

        pointers = tuple(
            0xA00000000 + index * 2 * probe.base.BLOCK_BYTES
            for index in range(shape.entry_count)
        )
        entries = probe.build_entries(
            pointers,
            layout="sparse-experts",
            logical_layer=2,
            bank_id=0,
            reuse_required=True,
            expert_ids=shape.expert_ids,
        )
        owners = tuple(int(entry[3]) for entry in entries)
        owner_base = 0x10000 * 3
        for expert_index, expert_id in enumerate(expert_ids):
            begin = expert_index * probe.ENVELOPES_PER_EXPERT
            end = begin + probe.ENVELOPES_PER_EXPERT
            self.assertEqual(
                owners[begin:end],
                (owner_base + expert_id,) * probe.ENVELOPES_PER_EXPERT,
            )
        self.assertEqual(tuple(int(entry[4]) for entry in entries), tuple(range(1, 46)))

        metrics = probe.layout_metrics(entries)
        probe.require_layout_contract(
            "sparse-experts",
            metrics,
            expected_entry_count=45,
            expected_logical_bytes=45 * probe.base.SLICE_BYTES,
        )
        self.assertEqual(metrics["partial_2m_blocks"], 45)
        self.assertEqual(metrics["adjacent_range_pairs"], 0)

    def test_sparse_partial64k_pte_contract(self) -> None:
        metrics = {
            "entry_count": 45,
            "logical_bytes": 45 * probe.base.SLICE_BYTES,
            "va_block_count": 45,
            "complete_2m_blocks": 0,
            "partial_2m_blocks": 45,
        }
        partial = probe.pte_contract("partial64k", metrics)
        self.assertEqual(
            partial,
            {
                "touched_2m_blocks": 45,
                "full_2m_blocks": 0,
                "partial_2m_blocks": 45,
                "active_64k_leaf_entries": 360,
                "active_4k_leaf_entries": 0,
                "activated_child_table_leaf_capacity": 1440,
                "active_envelopes": 45,
            },
        )
        fallback = probe.pte_contract("4k", metrics)
        self.assertEqual(fallback["active_64k_leaf_entries"], 0)
        self.assertEqual(fallback["active_4k_leaf_entries"], 45 * 128)
        self.assertEqual(fallback["activated_child_table_leaf_capacity"], 0)

    def test_complete_and_partial_64k_have_same_active_payload(self) -> None:
        complete_metrics = {
            "entry_count": probe.ENTRY_COUNT,
            "logical_bytes": probe.LOGICAL_BYTES,
            "va_block_count": probe.PACKED_ALLOCATION_COUNT,
            "complete_2m_blocks": probe.PACKED_ALLOCATION_COUNT,
            "partial_2m_blocks": 0,
        }
        partial_metrics = dict(
            complete_metrics,
            va_block_count=probe.ENTRY_COUNT,
            complete_2m_blocks=0,
            partial_2m_blocks=probe.ENTRY_COUNT,
        )
        complete = probe.pte_contract("64k", complete_metrics)
        partial = probe.pte_contract("partial64k", partial_metrics)
        self.assertEqual(complete["active_64k_leaf_entries"], 9216)
        self.assertEqual(partial["active_64k_leaf_entries"], 9216)
        self.assertEqual(complete["activated_child_table_leaf_capacity"], 9216)
        self.assertEqual(partial["activated_child_table_leaf_capacity"], 36864)

    def test_partial64k_validation_rejects_silent_4k_fallback(self) -> None:
        entry_count = 45
        logical_bytes = entry_count * probe.base.SLICE_BYTES
        contract = {
            "touched_2m_blocks": entry_count,
            "full_2m_blocks": 0,
            "partial_2m_blocks": entry_count,
            "active_64k_leaf_entries": entry_count * 8,
            "active_4k_leaf_entries": 0,
            "activated_child_table_leaf_capacity": entry_count * 32,
            "active_envelopes": entry_count,
        }
        record = {
            "_raw_result": {
                "ok": True,
                "plan_id": 10,
                "layer_id": 3,
                "generation": 7,
                "new_generation": 8,
                "mode": probe.base.MODE_TRANSITION,
                "requested_count": entry_count,
                "success_count": entry_count,
                "failure_count": 0,
                "requested_bytes": logical_bytes,
                "accepted_bytes": logical_bytes,
                "remap_count": entry_count,
                "ioctl_ret": 0,
                "rm_status": 0,
                "poisoned": False,
                "plan_execute": True,
                "copy_push_count": 1,
            },
            "latency_stats_delta": {
                "slot_plan_execute_calls": 1,
                "slot_direct_ioctl_calls": 1,
                "slot_direct_successes": 1,
                "slot_direct_errors": 0,
                "slot_direct_fallbacks": 0,
                "slot_direct_requested_ranges": entry_count,
                "slot_direct_completed_ranges": entry_count,
                "slot_direct_requested_bytes": logical_bytes,
                "slot_direct_completed_bytes": logical_bytes,
                "slot_direct_dual_bank_copy_first_calls": 1,
                "slot_direct_pte_4k_blocks": 0,
                "slot_direct_pte_64k_blocks": entry_count,
                "slot_direct_partial_64k_blocks": entry_count,
                "slot_direct_pte_write_entries": entry_count * 8,
                "slot_direct_shadow_pte_prefill_blocks": 0,
            },
        }
        probe.validate_execute(
            record,
            plan_id=10,
            layer_id=3,
            generation=7,
            initial_mode=probe.DEFAULT_PARTIAL_64K_INITIAL_MODE,
            initial=False,
            dual_bank_event=True,
            entry_count=entry_count,
            logical_bytes=logical_bytes,
            pte_mode="partial64k",
            expected_pte=contract,
            shadow_requested=True,
        )
        self.assertEqual(record["pte_observed"]["active_64k_leaf_entries"], 360)
        self.assertEqual(record["pte_observed"]["partial_64k_blocks"], 45)

        fallback = {
            **record,
            "latency_stats_delta": {
                **record["latency_stats_delta"],
                "slot_direct_pte_4k_blocks": entry_count,
                "slot_direct_pte_64k_blocks": 0,
                "slot_direct_pte_write_entries": entry_count * 128,
            },
        }
        with self.assertRaises(probe.ProbeError):
            probe.validate_execute(
                fallback,
                plan_id=10,
                layer_id=3,
                generation=7,
                initial_mode=probe.DEFAULT_PARTIAL_64K_INITIAL_MODE,
                initial=False,
                dual_bank_event=True,
                entry_count=entry_count,
                logical_bytes=logical_bytes,
                pte_mode="partial64k",
                expected_pte=contract,
                shadow_requested=True,
            )

        wrong_partial_count = {
            **record,
            "latency_stats_delta": {
                **record["latency_stats_delta"],
                "slot_direct_partial_64k_blocks": entry_count - 1,
            },
        }
        with self.assertRaises(probe.ProbeError):
            probe.validate_execute(
                wrong_partial_count,
                plan_id=10,
                layer_id=3,
                generation=7,
                initial_mode=probe.DEFAULT_PARTIAL_64K_INITIAL_MODE,
                initial=False,
                dual_bank_event=True,
                entry_count=entry_count,
                logical_bytes=logical_bytes,
                pte_mode="partial64k",
                expected_pte=contract,
                shadow_requested=True,
            )

    def test_partial64k_registration_requires_exact_prewarm_telemetry(self) -> None:
        entry_count = 45
        logical_bytes = entry_count * probe.base.SLICE_BYTES
        contract = {
            "touched_2m_blocks": entry_count,
            "full_2m_blocks": 0,
            "partial_2m_blocks": entry_count,
            "active_64k_leaf_entries": entry_count * 8,
            "active_4k_leaf_entries": 0,
            "activated_child_table_leaf_capacity": entry_count * 32,
            "active_envelopes": entry_count,
        }
        record = {
            "latency_stats_delta": {
                "slot_plan_pte_prewarm_calls": 1,
                "slot_plan_pte_prewarm_successes": 1,
                "slot_plan_pte_prewarm_failures": 0,
                "slot_plan_pte_prewarm_ranges": entry_count,
                "slot_plan_pte_prewarm_blocks": entry_count,
                "slot_plan_pte_prewarm_allocated_blocks": entry_count - 5,
                "slot_plan_pte_prewarm_reused_blocks": 5,
                "slot_plan_pte_prewarm_full_blocks": 0,
                "slot_plan_pte_prewarm_partial_blocks": entry_count,
                "slot_plan_pte_prewarm_bytes": logical_bytes,
            }
        }
        probe.validate_registration_prewarm(
            record,
            pte_mode="partial64k",
            entry_count=entry_count,
            logical_bytes=logical_bytes,
            expected_pte=contract,
        )
        self.assertEqual(record["pte_prewarm_observed"]["partial_blocks"], 45)
        self.assertEqual(record["pte_prewarm_observed"]["reused_blocks"], 5)

        missing_partial = {
            "latency_stats_delta": {
                **record["latency_stats_delta"],
                "slot_plan_pte_prewarm_partial_blocks": 0,
            }
        }
        with self.assertRaises(probe.ProbeError):
            probe.validate_registration_prewarm(
                missing_partial,
                pte_mode="partial64k",
                entry_count=entry_count,
                logical_bytes=logical_bytes,
                expected_pte=contract,
            )


if __name__ == "__main__":
    unittest.main()
