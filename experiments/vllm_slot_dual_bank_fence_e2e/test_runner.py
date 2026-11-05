#!/usr/bin/env python3
"""CPU-only tests for the dual-bank fence E2E runner and verifier."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import verify_case


SCRIPT_DIR = Path(__file__).resolve().parent


class VerifierTest(unittest.TestCase):
    def make_case(
        self,
        root: Path,
        *,
        fence: int,
        recipe: int,
        fast_validation: int = 0,
        shadow_pte_prefill: int = 0,
        feature_contract: bool = False,
    ) -> None:
        (root / "inputs").mkdir(parents=True)
        cold_keys = {
            (layer, expert)
            for layer in range(4)
            for expert in range(128)
        }
        all_keys = {
            (layer, expert)
            for layer in range(48)
            for expert in range(128)
        }
        plan = {
            "metadata": {
                "actual_offload_main_bytes": 4 * verify_case.LAYER_BYTES,
            },
            "offload_plan": [
                {"layer_id": layer, "expert_id": expert}
                for layer, expert in sorted(cold_keys)
            ],
            "hot_plan": [
                {"layer_id": layer, "expert_id": expert}
                for layer, expert in sorted(all_keys - cold_keys)
            ],
        }
        (root / "inputs" / "exact_l0_l3_plan.json").write_text(
            json.dumps(plan), encoding="utf-8"
        )

        result = {
            "status": "ok",
            "config": {"sequential_requests": 3},
            "execution": {
                "generate_call_count": 3,
                "single_llm_instance": True,
                "engine_shutdown_explicit": True,
            },
            "measurement": {
                "completed_requests": 3,
                "request_latency_ms": {"p50": 123.0},
            },
        }
        (root / "result.json").write_text(json.dumps(result), encoding="utf-8")

        if feature_contract:
            run_config = {
                "schema": "vllm_slot_dual_bank_fence_e2e_run_v1",
                "fence": fence,
                "metadata_recipe": recipe,
                "overlap_copy_prepare": 0 if shadow_pte_prefill else 1,
                "fast_validation": fast_validation,
                "shadow_pte_prefill": shadow_pte_prefill,
            }
            (root / "run_config.json").write_text(
                json.dumps(run_config), encoding="utf-8"
            )
            module_state = {
                "loaded_srcversion": "TESTSRC",
                "custom_srcversion": "TESTSRC",
                "uvm_moe_slot_pte_remap_enable": "1",
                "uvm_moe_slot_pool_vidmem_budget_bytes": str(
                    verify_case.CACHE_BYTES
                ),
                "uvm_moe_slot_plan_overlap_copy_prepare": str(
                    0 if shadow_pte_prefill else 1
                ),
                "uvm_moe_slot_plan_register_pte_prewarm": "1",
                "uvm_moe_slot_plan_cache_copy_runs": "1",
                "uvm_moe_slot_transition_fast_index": "1",
                "uvm_moe_slot_transition_direct_cutover": "1",
                "uvm_moe_slot_plan_fast_validation": str(fast_validation),
                "uvm_moe_slot_plan_shadow_pte_prefill": str(
                    shadow_pte_prefill
                ),
                "uvm_moe_slot_plan_cache_metadata_recipe": str(recipe),
                "uvm_moe_slot_plan_dual_bank_event": "1",
                "uvm_moe_slot_plan_cuda_fence": str(fence),
            }
            rendered_state = "".join(
                f"{key}={value}\n" for key, value in module_state.items()
            )
            for name in ("before", "after"):
                (root / f"module_state.{name}.env").write_text(
                    rendered_state, encoding="utf-8"
                )

        environment = {
            "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": "slot_restore",
            "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD": "1",
            "VLLM_UVM_SLOT_DIRECT_INIT_BANK": "1",
            "VLLM_UVM_SLOT_DIRECT_REUSE_BANK": "1",
            "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES": "1",
            "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION": "1",
            "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH": "1",
            "VLLM_UVM_SLOT_DIRECT_64K_PTE": "1",
            "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN": "1",
            "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT": "1",
            "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE": str(fence),
            "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB": str(
                (
                    verify_case.REPO_ROOT
                    / "workloads/vllm/vllm/uvm_test/"
                    "libuvm_slot_cuda_fence_kernels.so"
                ).resolve()
            ),
            "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES": str(verify_case.CACHE_BYTES),
            "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS": str(verify_case.MAX_SLOTS),
        }
        (root / "environment.txt").write_text(
            "".join(f"{key}={value}\n" for key, value in environment.items()),
            encoding="utf-8",
        )

        records = [
            {
                "operation": "register_layer_plans",
                "ok": True,
                "layer_count": 4,
                "bank_count": 2,
                "dual_bank_event": True,
                "requested_bytes_per_layer": {
                    str(layer): verify_case.LAYER_BYTES for layer in range(4)
                },
            }
        ]
        restore_index = 0
        for _request in range(3):
            for layer in range(4):
                low_level = {
                    "ok": True,
                    "cuda_fence_used": bool(fence),
                    "cuda_fence_accepted": bool(fence),
                    "cuda_completion_signaled": bool(fence),
                    "ioctl_wall_ns": 14_000_000 + layer,
                }
                if feature_contract:
                    low_level.update(
                        {
                            "plan_execute": True,
                            "cuda_stream_wait_enqueued": bool(fence),
                            "requested_bytes": verify_case.LAYER_BYTES,
                            "accepted_bytes": verify_case.LAYER_BYTES,
                            "success_count": verify_case.ENTRIES_PER_LAYER,
                            "remap_count": verify_case.ENTRIES_PER_LAYER,
                            "failure_count": 0,
                            "conflict_count": 0,
                            "poisoned": False,
                            "owner_cutover": restore_index > 0,
                            "generation": 100 + restore_index,
                            "new_generation": 101 + restore_index,
                        }
                    )
                records.append(
                    {
                        "operation": "restore_layer",
                        "ok": True,
                        "registered_plan_used": True,
                        "dual_bank_event": True,
                        "layer_id": layer,
                        "restore": low_level,
                    }
                )
                restore_index += 1
        records.append({"operation": "release", "ok": True})
        (root / "slot_trace.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        before = {
            "slot_plan_register_calls": 10,
            "slot_plan_execute_calls": 20,
            "slot_direct_copy_run_cache_hits": 30,
            "slot_direct_metadata_recipe_cache_attempts": 40,
            "slot_direct_target_recipe_cache_hits": 50,
            "slot_direct_clear_recipe_cache_hits": 60,
        }
        increments = {
            "slot_plan_register_calls": 5,
            "slot_plan_execute_calls": 12,
            "slot_direct_copy_run_cache_hits": 12,
            "slot_direct_metadata_recipe_cache_attempts": 12 if recipe else 0,
            "slot_direct_target_recipe_cache_hits": 12 if recipe else 0,
            "slot_direct_clear_recipe_cache_hits": 10 if recipe else 0,
        }
        if feature_contract:
            transitions = 11
            increments.update(
                {
                    "slot_direct_ioctl_calls": 12,
                    "slot_direct_successes": 12,
                    "slot_direct_errors": 0,
                    "slot_direct_dual_bank_copy_first_calls": transitions,
                    "slot_direct_transition_fast_index_calls": transitions,
                    "slot_direct_transition_group_clear_calls": transitions,
                    "slot_direct_transition_group_clear_fallbacks": 0,
                    "slot_direct_transition_direct_cutover_calls": transitions,
                    "slot_direct_shadow_pte_prefill_calls": (
                        transitions if shadow_pte_prefill else 0
                    ),
                    "slot_direct_shadow_pte_prefill_blocks": (
                        transitions * verify_case.BLOCKS_PER_LAYER
                        if shadow_pte_prefill
                        else 0
                    ),
                }
            )
            for key in increments:
                before.setdefault(key, 10)
        after = {key: value + increments[key] for key, value in before.items()}
        for name, values in (("before", before), ("after", after)):
            profile_calls_base = 5
            profile_ns_base = 1_000
            profile_bytes_base = 2_000
            if name == "after" and shadow_pte_prefill:
                profile_calls = profile_calls_base + 11
                profile_ns = profile_ns_base + 110_000
                profile_bytes = (
                    profile_bytes_base + 11 * verify_case.LAYER_BYTES
                )
            else:
                profile_calls = profile_calls_base
                profile_ns = profile_ns_base
                profile_bytes = profile_bytes_base
            profile = ""
            if feature_contract:
                profile = (
                    "slot_direct_profile:\n"
                    f"  shadow_pte_prefill calls={profile_calls} "
                    f"ns={profile_ns} avg_ns=1 pages=0 bytes={profile_bytes}\n"
                    f"  shadow_pte_publish calls={profile_calls} "
                    f"ns={profile_ns} avg_ns=1 pages=0 bytes={profile_bytes}\n"
                )
            (root / f"latency_stats.{name}.txt").write_text(
                profile
                + "".join(f"{key} {value}\n" for key, value in values.items()),
                encoding="utf-8",
            )

    def test_fenced_recipe_case_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_case(root, fence=1, recipe=1)
            summary = verify_case.verify_case(root, fence=1, metadata_recipe=1)
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["restore_cycles"], 3)
            self.assertEqual(summary["fence_accepted_count"], 12)
            self.assertGreater(
                summary["latency_stats_delta"][
                    "slot_direct_clear_recipe_cache_hits"
                ],
                0,
            )

    def test_unfenced_recipe_off_case_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_case(root, fence=0, recipe=0)
            summary = verify_case.verify_case(root, fence=0, metadata_recipe=0)
            self.assertEqual(summary["fence_used_count"], 0)
            self.assertEqual(
                summary["latency_stats_delta"][
                    "slot_direct_metadata_recipe_cache_attempts"
                ],
                0,
            )

    def test_fence_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_case(root, fence=0, recipe=1)
            with self.assertRaisesRegex(
                verify_case.VerificationError, "environment mismatch"
            ):
                verify_case.verify_case(root, fence=1, metadata_recipe=1)

    def test_fast_shadow_feature_contract_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_case(
                root,
                fence=1,
                recipe=1,
                fast_validation=1,
                shadow_pte_prefill=1,
                feature_contract=True,
            )
            summary = verify_case.verify_case(
                root,
                fence=1,
                metadata_recipe=1,
                fast_validation=1,
                shadow_pte_prefill=1,
            )
            self.assertTrue(summary["feature_contract"])
            self.assertEqual(summary["non_bootstrap_transition_count"], 11)
            self.assertEqual(
                summary["latency_stats_delta"][
                    "slot_direct_shadow_pte_prefill_blocks"
                ],
                11 * verify_case.BLOCKS_PER_LAYER,
            )

    def test_fast_shadow_bad_block_coverage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_case(
                root,
                fence=1,
                recipe=1,
                fast_validation=1,
                shadow_pte_prefill=1,
                feature_contract=True,
            )
            after = root / "latency_stats.after.txt"
            text = after.read_text(encoding="utf-8")
            expected = 10 + 11 * verify_case.BLOCKS_PER_LAYER
            after.write_text(
                text.replace(
                    f"slot_direct_shadow_pte_prefill_blocks {expected}",
                    f"slot_direct_shadow_pte_prefill_blocks {expected - 1}",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                verify_case.VerificationError, "did not cover all 288 blocks"
            ):
                verify_case.verify_case(
                    root,
                    fence=1,
                    metadata_recipe=1,
                    fast_validation=1,
                    shadow_pte_prefill=1,
                )

    def test_archived_artifact_cannot_claim_fast_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_case(root, fence=1, recipe=1)
            with self.assertRaisesRegex(
                verify_case.VerificationError, "predates fast/shadow telemetry"
            ):
                verify_case.verify_case(
                    root,
                    fence=1,
                    metadata_recipe=1,
                    fast_validation=1,
                    shadow_pte_prefill=1,
                )


class RunnerSourceTest(unittest.TestCase):
    def test_runner_has_fixed_e2e_contract_and_no_module_mutation(self) -> None:
        source = (SCRIPT_DIR / "run_case.sh").read_text(encoding="utf-8")
        self.assertIn("SLOT_CACHE_BYTES=1207959552", source)
        self.assertIn("SLOT_CACHE_MAX_SLOTS=2304", source)
        self.assertIn("SEQUENTIAL_REQUESTS=3", source)
        self.assertIn("uvm_moe_slot_plan_dual_bank_event 1", source)
        self.assertIn('uvm_moe_slot_plan_cuda_fence "${fence}"', source)
        self.assertIn("uvm_moe_slot_transition_fast_index 1", source)
        self.assertIn("uvm_moe_slot_transition_direct_cutover 1", source)
        self.assertIn(
            'uvm_moe_slot_plan_fast_validation "${fast_validation}"', source
        )
        self.assertIn(
            'uvm_moe_slot_plan_shadow_pte_prefill "${shadow_pte_prefill}"',
            source,
        )
        self.assertNotIn("modprobe ", source)
        self.assertNotIn("rmmod ", source)
        self.assertNotIn("insmod ", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
