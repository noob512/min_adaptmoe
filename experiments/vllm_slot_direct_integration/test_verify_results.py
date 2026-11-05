#!/usr/bin/env python3
"""CPU-only synthetic tests for the strict integration verifier."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from verify_results import VerificationError, verify_run


COUNTERS = (
    "slot_direct_ioctl_calls",
    "slot_direct_per_block_calls",
    "slot_direct_successes",
    "slot_direct_requested_ranges",
    "slot_direct_completed_ranges",
    "slot_direct_requested_bytes",
    "slot_direct_completed_bytes",
    "slot_direct_copy_pushes",
    "slot_direct_fallbacks",
    "slot_direct_errors",
    "slot_direct_vidmem_allocations",
    "slot_direct_vidmem_evictions",
    "slot_direct_pte_64k_blocks",
    "slot_direct_pte_2m_blocks",
)


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def stats_text(values: dict[str, int] | None = None) -> str:
    values = values or {}
    return "".join(f"{key} {values.get(key, 0)}\n" for key in COUNTERS)


def mode_result(mode: int, *, wall_ns: int, ioctl_ns: int) -> dict[str, Any]:
    remaps = 0 if mode == 8 else 1
    tlb = 0 if mode == 8 else 1
    pushes = 0 if mode == 8 else 1
    batch = {
        "ok": True,
        "reason": "ok",
        "problems": [],
        "input_indices": [0],
        "count": 1,
        "success_count": 1,
        "failure_count": 0,
        "conflict_count": 0,
        "requested_bytes": 4096,
        "accepted_bytes": 4096,
        "remap_count": remaps,
        "tlb_invalidate_count": tlb,
        "copy_push_count": pushes,
        "ioctl_ret": 0,
        "errno": 0,
        "rm_status": 0,
        "entry_statuses": [0],
        "wall_ns": ioctl_ns,
    }
    return {
        "ok": True,
        "reason": "ok",
        "error": "",
        "mode": mode,
        "requested_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "conflict_count": 0,
        "requested_bytes": 4096,
        "accepted_bytes": 4096,
        "remap_count": remaps,
        "tlb_invalidate_count": tlb,
        "copy_push_count": pushes,
        "ioctl_calls": 1,
        "ioctl_wall_ns": ioctl_ns,
        "wall_ns": wall_ns,
        "errno": 0,
        "rm_status": 0,
        "entry_statuses": [0],
        "batches": [batch],
    }


def release_result(*, wall_ns: int = 900, ioctl_ns: int = 700) -> dict[str, Any]:
    batch = {
        "ok": True,
        "reason": "ok",
        "problems": [],
        "input_indices": [0],
        "count": 1,
        "success_count": 1,
        "failure_count": 0,
        "conflict_count": 0,
        "requested_bytes": 0,
        "accepted_bytes": 4096,
        "remap_count": 1,
        "tlb_invalidate_count": 1,
        "copy_push_count": 0,
        "ioctl_ret": 0,
        "errno": 0,
        "rm_status": 0,
        "entry_statuses": [0],
        "wall_ns": ioctl_ns,
    }
    return {
        "ok": True,
        "reason": "ok",
        "error": "",
        "mode": 19,
        "requested_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "conflict_count": 0,
        "requested_bytes": 0,
        "accepted_bytes": 4096,
        "remap_count": 1,
        "tlb_invalidate_count": 1,
        "copy_push_count": 0,
        "ioctl_calls": 1,
        "ioctl_wall_ns": ioctl_ns,
        "wall_ns": wall_ns,
        "errno": 0,
        "rm_status": 0,
        "entry_statuses": [0],
        "batches": [batch],
    }


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture = self.root / "fixture.json"
        dump_json(
            fixture,
            {
                "metadata": {"prompt_token_ids_frozen": True},
                "requests": [
                    {
                        "request": 0,
                        "prompt": "synthetic",
                        "prompt_tokens": 2,
                        "prompt_token_ids": [1, 2],
                    }
                ],
            },
        )
        plan = self.root / "plan.json"
        all_coordinates = [
            {"layer_id": layer, "expert_id": expert}
            for layer in range(48)
            for expert in range(128)
        ]
        dump_json(
            plan,
            {
                "offload_plan": [all_coordinates[0]],
                "hot_plan": all_coordinates[1:],
                "metadata": {"actual_offload_main_bytes": 4096},
            },
        )
        config = {
            "schema": "vllm_slot_direct_integration_run_v1",
            "order": "AB",
            "arm_dirs": {"generic": "01_generic", "slot": "02_slot"},
            "benchmark": {
                "evaluation_fixture": str(fixture),
                "plan": str(plan),
                "concurrency": 1,
                "output_tokens": 4,
                "warmup_requests": 0,
            },
            "inputs": {
                "evaluation_fixture_sha256": sha256(fixture),
                "plan_sha256": sha256(plan),
            },
            "module": {
                "custom_srcversion": "CUSTOM",
                "loaded_srcversion": "CUSTOM",
                "slot_enable": 1,
                "slot_budget_bytes": 5 * 1024**3 // 2,
            },
        }
        dump_json(self.root / "run_config.json", config)
        for arm, directory in (("generic", "01_generic"), ("slot", "02_slot")):
            arm_dir = self.root / directory
            arm_dir.mkdir()
            (arm_dir / "returncode.txt").write_text("0\n", encoding="utf-8")
            for name in (
                "command.txt",
                "run.log",
                "execution.env",
                "gpu.before.csv",
                "gpu.after.csv",
            ):
                (arm_dir / name).write_text("synthetic\n", encoding="utf-8")
            trace_path = (arm_dir / "slot_trace.jsonl").resolve()
            access = "restore" if arm == "generic" else "slot_restore"
            (arm_dir / "environment.txt").write_text(
                f"VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE={access}\n"
                f"VLLM_UVM_SLOT_DIRECT_TRACE_FILE={trace_path}\n"
                f"VLLM_UVM_INIT_WEIGHT_PLAN_FILE={plan.resolve()}\n"
                "VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=\n",
                encoding="utf-8",
            )
            output_ids = [7, 8, 9, 10]
            output_hash = sha256_json(output_ids)
            aggregate_hash = sha256_json([output_ids])
            result = {
                "status": "ok",
                "profile_label": f"{arm}_restore",
                "mechanism_class": (
                    "vllm_generic_uvm_restore"
                    if arm == "generic"
                    else "vllm_slot_direct_mode8_mode10_restore"
                ),
                "measurement_start_epoch": 100.0,
                "measurement_end_epoch": 102.0,
                "config": {
                    "concurrency": 1,
                    "output_tokens": 4,
                    "warmup_requests": 0,
                },
                "fixtures": {"evaluation_sha256": sha256(fixture)},
                "environment": {
                    "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": access,
                    "VLLM_UVM_SLOT_DIRECT_TRACE_FILE": str(trace_path),
                    "VLLM_UVM_INIT_WEIGHT_PLAN_FILE": str(plan.resolve()),
                    "VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST": "",
                },
                "measurement": {
                    "elapsed_seconds": 0.100,
                    "completed_requests": 1,
                    "output_tokens": 4,
                    "output_tokens_per_second": 40.0,
                    "aggregate_output_token_ids_sha256": aggregate_hash,
                    "ttft_ms": {"count": 1, "p50": 10.0},
                    "request_latency_ms": {"count": 1, "p50": 100.0},
                    "requests": [
                        {
                            "fixture_request": 0,
                            "expected_prompt_tokens": 2,
                            "expected_prompt_token_ids_sha256": sha256_json([1, 2]),
                            "prompt_tokens": 2,
                            "prompt_token_ids_sha256": sha256_json([1, 2]),
                            "prompt_token_ids_exact": True,
                            "output_tokens": 4,
                            "output_token_ids": output_ids,
                            "output_token_ids_sha256": output_hash,
                            "decoded_text": "synthetic output",
                            "finish_reason": "length",
                            "ttft_ms": 10.0,
                            "tpot_ms": 30.0,
                            "request_latency_ms": 100.0,
                        }
                    ],
                },
            }
            dump_json(arm_dir / "result.json", result)
            (arm_dir / "latency_stats.before.txt").write_text(
                stats_text(), encoding="utf-8"
            )
            after = {}
            if arm == "slot":
                after = {
                    "slot_direct_ioctl_calls": 1,
                    "slot_direct_per_block_calls": 1,
                    "slot_direct_successes": 1,
                    "slot_direct_requested_ranges": 1,
                    "slot_direct_completed_ranges": 1,
                    "slot_direct_requested_bytes": 4096,
                    "slot_direct_completed_bytes": 4096,
                    "slot_direct_copy_pushes": 1,
                }
            (arm_dir / "latency_stats.after.txt").write_text(
                stats_text(after), encoding="utf-8"
            )
            if arm == "generic":
                (arm_dir / "slot_trace.jsonl").write_text("", encoding="utf-8")
            else:
                trace = {
                    "timestamp": 101.0,
                    "operation": "restore",
                    "layer_id": 0,
                    "expert_id": 0,
                    "ok": True,
                    "reason": "ok",
                    "statuses": {"region-0": True},
                    "failed_region_ids": [],
                    "logical_placement": "slot_gpu",
                    "ordinary_uvm_placement": "cpu",
                    "requested_bytes": 4096,
                    "slot_ids": {"region-0": 1},
                    "preallocate": mode_result(8, wall_ns=1000, ioctl_ns=800),
                    "restore": mode_result(10, wall_ns=2000, ioctl_ns=1500),
                    "wall_ns": 4000,
                }
                (arm_dir / "slot_trace.jsonl").write_text(
                    json.dumps(trace) + "\n", encoding="utf-8"
                )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def configure_cache_pair(self) -> None:
        config_path = self.root / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["experiment_kind"] = "cache_reuse"
        config["benchmark"].update(
            {"concurrency": 1, "sequential_requests": 2}
        )
        config["slot_cache"] = {"bytes": 4096, "max_slots": 1}
        dump_json(config_path, config)

        for arm, directory in (("generic", "01_generic"), ("slot", "02_slot")):
            arm_dir = self.root / directory
            environment_path = arm_dir / "environment.txt"
            environment_path.write_text(
                environment_path.read_text(encoding="utf-8")
                + "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES=4096\n"
                + "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS=1\n",
                encoding="utf-8",
            )
            result_path = arm_dir / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["benchmark_schema"] = "vllm_slot_cache_sequential_v1"
            result["mechanism_class"] = (
                "vllm_generic_uvm_restore_sequential"
                if arm == "generic"
                else "vllm_bounded_slot_cache_mode8_mode10_mode19"
            )
            result["config"]["concurrency"] = 1
            result["config"]["sequential_requests"] = 2
            result["environment"]["VLLM_UVM_SLOT_DIRECT_CACHE_BYTES"] = "4096"
            result["environment"]["VLLM_UVM_SLOT_DIRECT_MAX_SLOTS"] = "1"
            first = result["measurement"]["requests"][0]
            first.update(
                {
                    "sequence_index": 0,
                    "process_pid": 1234,
                    "llm_instance_id": 5678,
                    "generate_wall_ms": 80.0,
                }
            )
            second = dict(first)
            second.update(
                {
                    "request_id": "1",
                    "sequence_index": 1,
                    "generate_wall_ms": 80.0,
                }
            )
            requests = [first, second]
            result["execution"] = {
                "single_llm_instance": True,
                "llm_load_count": 1,
                "generate_call_count": 2,
                "requests_per_generate": 1,
                "process_pid": 1234,
                "llm_instance_id": 5678,
                "engine_shutdown_explicit": True,
                "shutdown_started_epoch": 102.1,
                "shutdown_end_epoch": 102.2,
                "calls": [
                    {
                        "sequence_index": 0,
                        "process_pid": 1234,
                        "llm_instance_id": 5678,
                        "start_epoch": 100.1,
                        "end_epoch": 100.9,
                    },
                    {
                        "sequence_index": 1,
                        "process_pid": 1234,
                        "llm_instance_id": 5678,
                        "start_epoch": 101.0,
                        "end_epoch": 101.8,
                    },
                ],
            }
            measurement = result["measurement"]
            measurement.update(
                {
                    "elapsed_seconds": 0.2,
                    "completed_requests": 2,
                    "sequential_generate_calls": 2,
                    "output_tokens": 8,
                    "output_tokens_per_second": 40.0,
                    "aggregate_output_token_ids_sha256": sha256_json(
                        [record["output_token_ids"] for record in requests]
                    ),
                    "ttft_ms": {"count": 2, "p50": 10.0},
                    "request_latency_ms": {"count": 2, "p50": 100.0},
                    "requests": requests,
                }
            )
            dump_json(result_path, result)

        identity = {
            "layer_id": 0,
            "expert_id": 0,
            "region_ids": ["region-0"],
            "slot_ids": {"region-0": 1},
        }
        common_restore = {
            **identity,
            "operation": "restore",
            "ok": True,
            "reason": "ok",
            "statuses": {"region-0": True},
            "failed_region_ids": [],
            "logical_placement": "slot_gpu",
            "ordinary_uvm_placement": "cpu",
            "requested_bytes": 4096,
            "preallocate": mode_result(8, wall_ns=1000, ioctl_ns=800),
            "restore": mode_result(10, wall_ns=2000, ioctl_ns=1500),
            "wall_ns": 4000,
            "limits": {"bytes": 4096, "max_slots": 1},
        }
        records = [
            {
                **common_restore,
                "timestamp": 100.2,
                "cache_before": {"bytes": 0, "slots": 0},
                "cache_after": {"bytes": 4096, "slots": 1},
                "evictions": [],
            },
            {
                **identity,
                "timestamp": 100.8,
                "operation": "release",
                "trigger": "capacity",
                "ok": True,
                "reason": "ok",
                "result": release_result(),
                "cache_before": {"bytes": 4096, "slots": 1},
                "cache_after": {"bytes": 0, "slots": 0},
                "limits": {"bytes": 4096, "max_slots": 1},
            },
            {
                **common_restore,
                "timestamp": 101.2,
                "cache_before": {"bytes": 0, "slots": 0},
                "cache_after": {"bytes": 4096, "slots": 1},
                "evictions": [identity],
            },
            {
                "layer_id": -1,
                "expert_id": -1,
                "region_ids": ["region-0"],
                "slot_ids": {"region-0": 1},
                "timestamp": 102.1,
                "operation": "release",
                "trigger": "explicit",
                "scope": "session",
                "ok": True,
                "reason": "closed",
                "result": {
                    "ok": True,
                    "reason": "closed",
                    "closed": True,
                    "release": release_result(),
                    "wall_ns": 1200,
                },
                "cache_before": {"bytes": 4096, "slots": 1},
                "cache_after": {"bytes": 0, "slots": 0},
                "limits": {"bytes": 4096, "max_slots": 1},
            },
        ]
        slot_dir = self.root / "02_slot"
        (slot_dir / "slot_trace.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        slot_after = {
            "slot_direct_ioctl_calls": 2,
            "slot_direct_per_block_calls": 2,
            "slot_direct_successes": 2,
            "slot_direct_requested_ranges": 2,
            "slot_direct_completed_ranges": 2,
            "slot_direct_requested_bytes": 8192,
            "slot_direct_completed_bytes": 8192,
            "slot_direct_copy_pushes": 2,
        }
        (slot_dir / "latency_stats.after.txt").write_text(
            stats_text(slot_after), encoding="utf-8"
        )

    def test_complete_pair_passes(self) -> None:
        summary = verify_run(self.root)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["correctness"]["mode8_mode10_complete"])
        self.assertEqual(summary["slot_placement"]["requested_bytes"], 4096)

    def test_output_token_difference_fails(self) -> None:
        path = self.root / "02_slot" / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["measurement"]["requests"][0]["output_token_ids"][-1] = 11
        dump_json(path, result)
        with self.assertRaisesRegex(VerificationError, "output token"):
            verify_run(self.root)

    def test_driver_fallback_fails(self) -> None:
        path = self.root / "02_slot" / "latency_stats.after.txt"
        values = {
            "slot_direct_ioctl_calls": 1,
            "slot_direct_per_block_calls": 1,
            "slot_direct_successes": 1,
            "slot_direct_requested_ranges": 1,
            "slot_direct_completed_ranges": 1,
            "slot_direct_requested_bytes": 4096,
            "slot_direct_completed_bytes": 4096,
            "slot_direct_copy_pushes": 1,
            "slot_direct_fallbacks": 1,
        }
        path.write_text(stats_text(values), encoding="utf-8")
        with self.assertRaisesRegex(VerificationError, "fallbacks"):
            verify_run(self.root)

    def test_bounded_cache_release_reuse_passes(self) -> None:
        self.configure_cache_pair()
        summary = verify_run(self.root)
        self.assertEqual(summary["experiment_kind"], "cache_reuse")
        self.assertTrue(summary["slot_placement"]["release_then_restore_proven"])
        self.assertTrue(summary["slot_placement"]["final_cache_empty"])
        self.assertEqual(summary["slot_placement"]["capacity_releases"], 1)

    def test_cache_release_without_later_same_identity_fails(self) -> None:
        self.configure_cache_pair()
        path = self.root / "02_slot" / "slot_trace.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[2]["expert_id"] = 1
        records[2]["region_ids"] = ["region-1"]
        records[2]["slot_ids"] = {"region-1": 2}
        records[2]["statuses"] = {"region-1": True}
        records[2]["evictions"] = [records[0] | {
            "region_ids": ["region-0"],
            "slot_ids": {"region-0": 1},
        }]
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(VerificationError, "restored later"):
            verify_run(self.root)

    def test_cache_final_resource_leak_fails(self) -> None:
        self.configure_cache_pair()
        path = self.root / "02_slot" / "slot_trace.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[-1]["cache_after"] = {"bytes": 4096, "slots": 1}
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(VerificationError, "did not reduce|not empty"):
            verify_run(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
