from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.uvm_slot_shadow_epoch_probe import summarize


def make_artifact(
    steady_ioctl_ns: int = 200,
    *,
    srcversion: str = "TESTSRC",
    instrumented: bool = True,
    fast_validation: bool = True,
    shadow_pte_prefill: bool = True,
) -> dict:
    execution_order = [0, 1, 2, 3, 0]
    bank_order = [0, 1, 0, 1, 0]
    steady_values = [steady_ioctl_ns - 10, steady_ioctl_ns, steady_ioctl_ns + 10]
    ioctl_values = [9_000_000, 8_000_000, *steady_values]
    executions = []
    profiles = []
    barriers = []
    generation_start = 100
    execution_plan_ids = [1, 6, 10, 14, 2]
    for index, (layer_id, bank_id, ioctl_ns) in enumerate(
        zip(execution_order, bank_order, ioctl_values)
    ):
        executions.append(
            {
                "ok": True,
                "ioctl_ret": 0,
                "errno": 0,
                "reason": "ok",
                "layer_id": layer_id,
                "plan_id": execution_plan_ids[index],
                "bank_id": bank_id,
                "generation": generation_start + index,
                "new_generation": generation_start + index + 1,
                "compute_event_completed_before_execute": True,
                "success_count": 1152,
                "failure_count": 0,
                "accepted_bytes": 603_979_776,
                "ioctl_wall_ns": ioctl_ns,
                "wall_ns": ioctl_ns + 7,
            }
        )
        profile = {
            "slot_direct_profile.total.ns": ioctl_ns - 1,
            "slot_direct_profile.prepare.ns": ioctl_ns // 5,
            "slot_direct_profile.transition_clear_rebind.ns": ioctl_ns // 6,
            "slot_direct_profile.unmap.ns": ioctl_ns // 7,
            "slot_direct_profile.copy.ns": ioctl_ns // 8,
            "slot_direct_profile.map.ns": ioctl_ns // 9,
            "slot_direct_profile.wait.ns": ioctl_ns // 2,
            "slot_direct_profile.flatten_sort.ns": ioctl_ns // 10,
            "slot_direct_profile.slot_validate.ns": ioctl_ns // 11,
            "slot_direct_profile.transition_preflight.ns": ioctl_ns // 12,
            "slot_direct_profile.copy_complete.ns": 0,
            "slot_direct_profile.copy_complete.calls": 0,
            "slot_direct_profile.map_complete.ns": 0,
            "slot_direct_profile.map_complete.calls": 0,
            "slot_direct_copy_run_cache_hits": 1,
        }
        if instrumented:
            shadow_expected = shadow_pte_prefill and index > 0
            prefill_ns = ioctl_ns // 13 if shadow_expected else 0
            publish_ns = ioctl_ns // 14 if shadow_expected else 0
            profile.update(
                {
                    "slot_direct_shadow_pte_prefill_calls": (
                        1 if shadow_expected else 0
                    ),
                    "slot_direct_shadow_pte_prefill_blocks": (
                        288 if shadow_expected else 0
                    ),
                    "slot_direct_profile.shadow_pte_prefill.calls": (
                        1 if shadow_expected else 0
                    ),
                    "slot_direct_profile.shadow_pte_prefill.ns": prefill_ns,
                    "slot_direct_profile.shadow_pte_publish.calls": (
                        1 if shadow_expected else 0
                    ),
                    "slot_direct_profile.shadow_pte_publish.ns": publish_ns,
                }
            )
        profiles.append(profile)
        barriers.append(
            {
                "sequence_index": index,
                "layer_id": layer_id,
                "bank_id": bank_id,
                "checksum": index + 100,
                "expected": index + 100,
                "execute_end_ns": 10_000 + index,
                "completed_ns": 20_000 + index,
            }
        )
    module_parameters = {
        "dual_bank_event": 1,
        "copy_run_cache": 1,
        "pte_prewarm": 1,
        "legacy_overlap": 0,
        "fast_index": 1,
        "direct_cutover": 1,
        "vidmem_budget_bytes": 1_207_959_552,
        "shadow_epoch_cache": 1,
    }
    if instrumented:
        module_parameters.update(
            {
                "fast_validation": int(fast_validation),
                "shadow_pte_prefill": int(shadow_pte_prefill),
            }
        )
    plan_ids = [1, 2, 6, 10, 14]
    registration_layers = [0, 0, 1, 2, 3]
    final_generation = generation_start + len(execution_order)
    return {
        "status": "pass",
        "blocks": 288,
        "entries_per_bank": 1152,
        "bytes_per_bank": 603_979_776,
        "total_bank_bytes": 1_207_959_552,
        "execution_order": execution_order,
        "bank_order": bank_order,
        "compute_copy_overlap_allowed": False,
        "compute_event_waited_before_each_transition": True,
        "worker_returncode": 0,
        "gpu_memory_final_mib": 0,
        "module_srcversion": srcversion,
        "module_parameters": module_parameters,
        "registrations": [
            {
                "ok": True,
                "plan_id": plan_id,
                "layer_id": layer_id,
                "generation": generation_start,
            }
            for plan_id, layer_id in zip(plan_ids, registration_layers)
        ],
        "executions": executions,
        "execute_profile_deltas": profiles,
        "event_barriers": barriers,
        "destroy_results": [
            {
                "ok": True,
                "destroyed": True,
                "ioctl_ret": 0,
                "errno": 0,
                "plan_id": plan_id,
                "generation": final_generation,
                "current_generation": (
                    0 if index == len(plan_ids) - 1 else final_generation
                ),
            }
            for index, plan_id in enumerate(plan_ids)
        ],
        "close": {"ok": True, "reason": "closed"},
        "pre_close_state": {
            "allocated_slots": 2304,
            "plan_count": 0,
            "session_open": True,
        },
        "post_close_state": {
            "allocated_slots": 0,
            "plan_count": 0,
            "session_open": False,
        },
    }


class SummarizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_round(self, name: str, artifact: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        return path

    def write_arm(self, prefix: str, medians: list[int], srcversion: str) -> list[Path]:
        return [
            self.write_round(
                f"{prefix}_r{index}.json",
                make_artifact(median, srcversion=srcversion),
            )
            for index, median in enumerate(medians, start=1)
        ]

    def test_paired_summary_uses_only_bank_reuse_samples(self) -> None:
        reference = self.write_arm("safe", [200, 220, 240], "SAFE")
        candidate = self.write_arm("candidate", [180, 200, 220], "CANDIDATE")
        result = summarize.build_summary(
            "safe",
            reference,
            "candidate",
            candidate,
            expect_b={"shadow_epoch_cache": 1},
        )

        reference_round = result["arms"]["reference"]["rounds"][0]
        self.assertEqual(reference_round["steady_indices"], [2, 3, 4])
        self.assertEqual(
            reference_round["steady_samples_ns"]["ioctl_wall_ns"],
            [190, 200, 210],
        )
        self.assertEqual(
            reference_round["steady_median_ns"]["ioctl_wall_ns"], 200
        )
        self.assertNotIn(
            9_000_000,
            reference_round["steady_samples_ns"]["ioctl_wall_ns"],
        )

        comparison = result["comparison"]["metrics"]["ioctl_wall_ns"]
        self.assertEqual(comparison["paired_delta_median_ns"], 20)
        self.assertEqual(comparison["candidate_wins"], 3)
        self.assertEqual(comparison["ties"], 0)
        self.assertEqual(comparison["candidate_losses"], 0)
        self.assertAlmostEqual(
            comparison["paired_rounds"][0]["speedup_reference_over_candidate"],
            200 / 180,
        )
        self.assertIn("copy_submit_ns", result["telemetry_contract"])
        self.assertIn("not device completion", result["telemetry_contract"]["copy_submit_ns"])
        self.assertIn(
            "shadow_pte_prefill_ns", result["arms"]["candidate"]["metrics"]
        )
        evidence = result["arms"]["candidate"]["rounds"][0]["feature_evidence"]
        self.assertTrue(evidence["fast_validation_enabled"])
        self.assertTrue(evidence["shadow_pte_prefill_enabled"])
        self.assertEqual(evidence["shadow_prefill_verified_steady_executions"], 3)
        self.assertEqual(evidence["shadow_prefill_blocks_per_steady_execution"], 288)

    def test_records_sha256_and_per_arm_srcversion(self) -> None:
        reference = self.write_arm("safe", [200, 200, 200], "SAFE")
        candidate = self.write_arm("candidate", [200, 200, 200], "CANDIDATE")
        result = summarize.build_summary(
            "safe", reference, "candidate", candidate
        )
        first = result["arms"]["reference"]["rounds"][0]
        expected_hash = hashlib.sha256(reference[0].read_bytes()).hexdigest()
        self.assertEqual(first["sha256"], expected_hash)
        self.assertEqual(result["arms"]["reference"]["module_srcversion"], "SAFE")
        self.assertEqual(
            result["arms"]["candidate"]["module_srcversion"], "CANDIDATE"
        )
        comparison = result["comparison"]["metrics"]["ioctl_wall_ns"]
        self.assertEqual(comparison["ties"], 3)

    def test_rejects_safety_and_cleanup_contract_violations(self) -> None:
        mutations = {
            "module parameter": lambda value: value["module_parameters"].__setitem__(
                "dual_bank_event", 0
            ),
            "execution order": lambda value: value.__setitem__(
                "execution_order", [0, 2, 1, 3, 0]
            ),
            "GPU leak": lambda value: value.__setitem__("gpu_memory_final_mib", 1),
            "checksum": lambda value: value["event_barriers"][2].__setitem__(
                "checksum", -1
            ),
            "slot leak": lambda value: value["post_close_state"].__setitem__(
                "allocated_slots", 1
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                artifact = make_artifact()
                mutate(artifact)
                with self.assertRaises(summarize.ValidationError):
                    summarize.validate_artifact(artifact, name)

    def test_rejects_nonzero_device_completion_pseudo_metrics(self) -> None:
        for key in (
            "slot_direct_profile.copy_complete.ns",
            "slot_direct_profile.map_complete.ns",
        ):
            with self.subTest(key=key):
                artifact = make_artifact()
                artifact["execute_profile_deltas"][2][key] = 1
                with self.assertRaisesRegex(
                    summarize.ValidationError, "submission timings"
                ):
                    summarize.validate_artifact(artifact, key)

    def test_rejects_round_count_mismatch(self) -> None:
        reference = self.write_arm("safe", [200, 200, 200], "SAFE")
        candidate = self.write_arm("candidate", [200, 200], "CANDIDATE")
        with self.assertRaisesRegex(summarize.ValidationError, "round counts differ"):
            summarize.build_summary("safe", reference, "candidate", candidate)

    def test_rejects_srcversion_change_within_arm(self) -> None:
        paths = self.write_arm("safe", [200, 200, 200], "SAFE")
        changed = make_artifact(200, srcversion="CHANGED")
        paths[2] = self.write_round("safe_changed.json", changed)
        with self.assertRaisesRegex(summarize.ValidationError, "changed within arm"):
            summarize.summarize_arm("safe", paths)

    def test_extra_parameter_expectation_is_fail_closed(self) -> None:
        artifact = make_artifact()
        summarize.validate_artifact(
            artifact, "candidate", {"shadow_epoch_cache": 1}
        )
        with self.assertRaisesRegex(summarize.ValidationError, "shadow_epoch_cache"):
            summarize.validate_artifact(
                artifact, "candidate", {"shadow_epoch_cache": 0}
            )

    def test_shadow_prefill_and_publish_must_hit_and_have_timing(self) -> None:
        mutations = {
            "prefill calls": lambda value: value["execute_profile_deltas"][2].__setitem__(
                "slot_direct_shadow_pte_prefill_calls", 0
            ),
            "prefill blocks": lambda value: value["execute_profile_deltas"][2].__setitem__(
                "slot_direct_shadow_pte_prefill_blocks", 287
            ),
            "prefill timing": lambda value: value["execute_profile_deltas"][2].__setitem__(
                "slot_direct_profile.shadow_pte_prefill.ns", 0
            ),
            "publish timing": lambda value: value["execute_profile_deltas"][2].__setitem__(
                "slot_direct_profile.shadow_pte_publish.ns", 0
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                artifact = make_artifact()
                mutate(artifact)
                with self.assertRaises(summarize.ValidationError):
                    summarize.validate_artifact(artifact, name)

    def test_shadow_disabled_rejects_unexpected_hit(self) -> None:
        artifact = make_artifact(shadow_pte_prefill=False)
        summarize.validate_artifact(artifact, "shadow off")
        artifact["execute_profile_deltas"][2][
            "slot_direct_shadow_pte_prefill_calls"
        ] = 1
        with self.assertRaisesRegex(summarize.ValidationError, "unexpected shadow"):
            summarize.validate_artifact(artifact, "shadow off")

    def test_generation_chain_and_final_destroy_are_verified(self) -> None:
        mutations = {
            "non-advancing generation": lambda value: value["executions"][2].__setitem__(
                "new_generation", value["executions"][2]["generation"]
            ),
            "broken generation chain": lambda value: value["executions"][3].__setitem__(
                "generation", value["executions"][3]["generation"] + 2
            ),
            "destroy generation": lambda value: value["destroy_results"][0].__setitem__(
                "generation", 1
            ),
            "session generation leak": lambda value: value["destroy_results"][-1].__setitem__(
                "current_generation", 105
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                artifact = make_artifact()
                mutate(artifact)
                with self.assertRaisesRegex(summarize.ValidationError, "generation"):
                    summarize.validate_artifact(artifact, name)

    def test_archived_telemetry_is_supported_but_cannot_mix_with_new(self) -> None:
        legacy = make_artifact(instrumented=False)
        summarize.validate_artifact(legacy, "legacy")
        legacy_paths = [
            self.write_round(f"legacy_r{index}.json", legacy)
            for index in range(1, 4)
        ]
        new_paths = self.write_arm("new", [200, 200, 200], "NEW")
        with self.assertRaisesRegex(summarize.ValidationError, "schemas differ"):
            summarize.build_summary(
                "legacy", legacy_paths, "new", new_paths
            )

    def test_rejects_feature_configuration_change_within_arm(self) -> None:
        paths = self.write_arm("mixed", [200, 200, 200], "MIXED")
        changed = make_artifact(200, srcversion="MIXED", shadow_pte_prefill=False)
        paths[2] = self.write_round("mixed_changed.json", changed)
        with self.assertRaisesRegex(summarize.ValidationError, "changed within arm"):
            summarize.summarize_arm("mixed", paths)

    def test_parse_expectations_supports_json_scalars(self) -> None:
        self.assertEqual(
            summarize.parse_expectations(
                ["shadow_epoch_cache=1", "feature=true", "name=candidate"]
            ),
            {"shadow_epoch_cache": 1, "feature": True, "name": "candidate"},
        )
        with self.assertRaises(summarize.ValidationError):
            summarize.parse_expectations(["missing_separator"])


if __name__ == "__main__":
    unittest.main()
