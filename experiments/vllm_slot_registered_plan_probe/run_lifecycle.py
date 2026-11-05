#!/usr/bin/env python3
"""Deterministic SIGTERM lifecycle probe for the registered-plan slot bank.

The controller launches one in-process vLLM child.  The child initializes the
fixed bank, completes exactly one frozen request, atomically publishes a round
marker, and then blocks without installing a SIGTERM handler.  The controller
signals the process group only after observing that marker and verifies the
driver counters, fd-close reclaim record, GPU cleanup, and generation recovery.

Importing this module is CPU-only.  torch and vLLM are imported exclusively by
the private ``child`` subcommand after the live-run gate has been checked by the
controller.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import importlib.util
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VLLM_ROOT = REPO_ROOT / "workloads" / "vllm" / "vllm"
VLLM_PY = VLLM_ROOT / ".venv" / "bin" / "python"
COMMON_BENCHMARK = (
    REPO_ROOT / "experiments" / "broad_offload_compare"
    / "benchmark_custom_offload.py"
)
PREPARE_INPUTS = (
    REPO_ROOT / "experiments" / "vllm_slot_vs_pinned_copy"
    / "prepare_inputs.py"
)
ABORT_PROBE = (
    REPO_ROOT / "experiments" / "uvm_slot_registered_plan_probe"
    / "run_abort_probe.py"
)
DEFAULT_LIBRARY = (
    REPO_ROOT / "experiments" / "uvm_512x4m_batch_compare"
    / "libuvm_batch_shim.so"
)
DEFAULT_MODEL = (
    REPO_ROOT.parent / ".cache" / "huggingface" / "hub"
    / "models--Qwen--Qwen3-30B-A3B-FP8" / "snapshots"
    / "d206ba732169f29bb77fbf80fc2c4b81d4d30782"
)
DEFAULT_FIXTURE = (
    REPO_ROOT / "experiments" / "eurosys_moe_offload_evaluation" / "inputs"
    / "legacy_pilot" / "qwen3_30b_a3b_fp8" / "fixtures"
    / "evaluation_fixture_token_ids.json"
)
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")
DEFAULT_ROUTE_LOG = Path("/proc/driver/nvidia-uvm/phase_f_route_log")
DEFAULT_ROUTE_ENABLE = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_enable"
)
DEFAULT_SRCVERSION = Path("/sys/module/nvidia_uvm/srcversion")
DEFAULT_PTE_ENABLE = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable"
)
DEFAULT_FAST_INDEX = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_fast_index"
)
DEFAULT_DIRECT_CUTOVER = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_direct_cutover"
)
DEFAULT_BUDGET = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes"
)
DEFAULT_MODULE = (
    REPO_ROOT / "kernel-module" / "nvidia-module" / "kernel-open"
    / "nvidia-uvm.ko"
)

BANK_SLOTS = 1152
BANK_BYTES = 603_979_776
LAYERS = 4
RANGES_PER_LAYER = 1152
BLOCKS_PER_LAYER = 288
PTE_PER_BLOCK = 32
PAGES_PER_LAYER = BANK_BYTES // 4096
EXPECTED_TOKEN_IDS = [4710]

_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*$")
_PROFILE_RE = re.compile(
    r"^\s+([a-z][a-z0-9_]*)\s+"
    r"calls=(\d+)\s+ns=(\d+)\s+avg_ns=(\d+)\s+"
    r"pages=(\d+)\s+bytes=(\d+)\s*$"
)
_COUNTER_RE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)\s+(0x[0-9a-fA-F]+|\d+)\s*$"
)
_ROUTE_RE = re.compile(
    r"^\[(\d+)\.(\d{1,9})\]\s+moe_slot_pool_close_reclaim\s+(.*)$"
)


class LifecycleError(RuntimeError):
    """A lifecycle invariant was not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    require(type(value) is int, f"{name} must be an integer")
    parsed = int(value)
    require(parsed >= minimum, f"{name} must be >= {minimum}")
    return parsed


def atomic_write_json(path: Path, value: object) -> None:
    """Write a marker atomically so the peer never observes partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def parse_latency_stats(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    section: str | None = None

    def add(name: str, value: int, line_number: int) -> None:
        require(name not in values, f"duplicate latency key {name} at {line_number}")
        values[name] = value

    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        section_match = _SECTION_RE.fullmatch(line)
        if section_match:
            section = section_match.group(1)
            continue
        profile_match = _PROFILE_RE.fullmatch(line)
        if profile_match:
            require(section is not None, f"profile row before section at {line_number}")
            name, calls, nanoseconds, _average, pages, bytes_ = profile_match.groups()
            prefix = f"{section}.{name}"
            add(f"{prefix}.calls", int(calls), line_number)
            add(f"{prefix}.ns", int(nanoseconds), line_number)
            add(f"{prefix}.pages", int(pages), line_number)
            add(f"{prefix}.bytes", int(bytes_), line_number)
            continue
        counter_match = _COUNTER_RE.fullmatch(line)
        if counter_match:
            name, value = counter_match.groups()
            add(name, int(value, 0), line_number)
            section = None
            continue
        raise LifecycleError(
            f"unrecognized latency_stats line {line_number}: {line!r}"
        )
    return values


def stats_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
    *,
    required: set[str] | frozenset[str] = frozenset(),
) -> dict[str, int]:
    for name in required:
        require(name in before, f"before snapshot lacks {name}")
        require(name in after, f"after snapshot lacks {name}")
    delta: dict[str, int] = {}
    for name in set(before) & set(after):
        value = int(after[name]) - int(before[name])
        require(value >= 0, f"driver counter moved backwards: {name}")
        delta[name] = value
    return delta


def expected_sigterm_delta(requests: int, candidate_ioctl_calls: int) -> dict[str, int]:
    requests = _strict_int(requests, "requests", minimum=1)
    candidate_ioctl_calls = _strict_int(
        candidate_ioctl_calls, "candidate_ioctl_calls", minimum=1
    )
    executes = LAYERS * requests
    transitions = executes - 1
    ranges = RANGES_PER_LAYER * executes
    blocks = BLOCKS_PER_LAYER * executes
    bytes_ = BANK_BYTES * executes
    expected = {
        "slot_plan_register_calls": 5,
        "slot_plan_execute_calls": executes,
        "slot_plan_destroy_calls": 0,
        "slot_session_abort_calls": 0,
        "slot_session_abort_successes": 0,
        "slot_session_abort_failures": 0,
        "slot_session_abort_released_slots": 0,
        "slot_session_abort_released_bytes": 0,
        "slot_session_abort_remaining_slots": 0,
        "slot_legacy_ioctl_calls": candidate_ioctl_calls,
        "slot_legacy_transition_calls": 0,
        "slot_direct_ioctl_calls": executes,
        "slot_direct_per_block_calls": 0,
        "slot_direct_shared_calls": executes,
        "slot_direct_successes": executes,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
        "slot_direct_requested_ranges": ranges,
        "slot_direct_completed_ranges": ranges,
        "slot_direct_requested_bytes": bytes_,
        "slot_direct_completed_bytes": bytes_,
        "slot_direct_blocks": blocks,
        "slot_direct_pte_4k_blocks": 0,
        "slot_direct_pte_64k_blocks": blocks,
        "slot_direct_pte_2m_blocks": 0,
        "slot_direct_pte_write_entries": blocks * PTE_PER_BLOCK,
        "slot_direct_skipped_unmap_blocks": 0,
        "slot_direct_stage_sync_calls": 0,
        "slot_direct_copy_pushes": executes,
        "slot_direct_memcopy_methods": ranges,
        "slot_direct_pte_map_ranges": ranges,
        "slot_direct_pte_map_bytes": bytes_,
        "slot_direct_tracker_waits": executes,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        "slot_direct_transition_fast_index_calls": transitions,
        "slot_direct_transition_group_clear_calls": transitions,
        "slot_direct_transition_group_clear_fallbacks": 0,
        "slot_direct_transition_clear_ranges": RANGES_PER_LAYER * transitions,
        "slot_direct_transition_clear_unique_blocks": BLOCKS_PER_LAYER * transitions,
        "slot_direct_transition_direct_cutover_calls": transitions,
        "slot_direct_transition_direct_cutover_pages": PAGES_PER_LAYER * transitions,
    }
    return expected


def verify_exact_delta(actual: Mapping[str, int], expected: Mapping[str, int]) -> None:
    for name, value in expected.items():
        require(name in actual, f"missing counter delta {name}")
        require(
            int(actual[name]) == value,
            f"{name}: expected {value}, observed {actual[name]}",
        )


def parse_close_reclaim_records(
    text: str,
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if "moe_slot_pool_close_reclaim" not in line:
            continue
        match = _ROUTE_RE.fullmatch(line.strip())
        require(match is not None, f"malformed close-reclaim line {line_number}")
        seconds, fraction, body = match.groups()
        timestamp_ns = int(seconds) * 1_000_000_000 + int(fraction.ljust(9, "0"))
        if start_ns is not None and timestamp_ns < start_ns:
            continue
        if end_ns is not None and timestamp_ns > end_ns:
            continue
        fields: dict[str, str] = {}
        for token in body.split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        required = (
            "va_space", "reclaimed", "bytes", "failed", "total_reclaimed",
            "total_bytes", "total_failed", "owner_mismatches",
        )
        for name in required:
            require(name in fields, f"close-reclaim line lacks {name}")
        record: dict[str, Any] = {
            "timestamp_ns": timestamp_ns,
            "va_space": fields["va_space"],
            "line": line,
        }
        for name in required[1:]:
            try:
                value = int(fields[name], 0)
            except ValueError as exc:
                raise LifecycleError(f"invalid close-reclaim {name}") from exc
            require(value >= 0, f"negative close-reclaim {name}")
            record[name] = value
        records.append(record)
    return records


def aggregate_close_reclaim(records: list[Mapping[str, Any]]) -> dict[str, int]:
    seen: set[tuple[object, ...]] = set()
    unique: list[Mapping[str, Any]] = []
    for record in records:
        identity = (
            record.get("timestamp_ns"), record.get("va_space"),
            record.get("total_reclaimed"), record.get("total_bytes"),
            record.get("total_failed"),
        )
        if identity not in seen:
            seen.add(identity)
            unique.append(record)
    return {
        "records": len(unique),
        "nonzero_records": sum(
            int(item["reclaimed"]) > 0 or int(item["failed"]) > 0
            for item in unique
        ),
        "reclaimed": sum(int(item["reclaimed"]) for item in unique),
        "bytes": sum(int(item["bytes"]) for item in unique),
        "failed": sum(int(item["failed"]) for item in unique),
    }


def verify_close_reclaim(records: list[Mapping[str, Any]]) -> dict[str, int]:
    aggregate = aggregate_close_reclaim(records)
    require(aggregate["nonzero_records"] >= 1, "missing nonzero close reclaim")
    require(aggregate["reclaimed"] == BANK_SLOTS, "close reclaim slot mismatch")
    require(aggregate["bytes"] == BANK_BYTES, "close reclaim byte mismatch")
    require(aggregate["failed"] == 0, "close reclaim reported failures")
    return aggregate


def _wait_close_reclaim_log(
    path: Path,
    *,
    start_ns: int,
    timeout: float,
) -> tuple[str, list[dict[str, Any]]]:
    """Wait for deferred file-release cleanup to publish its route record."""
    deadline = time.monotonic() + timeout
    text = ""
    records: list[dict[str, Any]] = []
    while True:
        text = path.read_text(encoding="utf-8")
        records = parse_close_reclaim_records(text, start_ns=start_ns)
        aggregate = aggregate_close_reclaim(records)
        if (
            aggregate["reclaimed"] == BANK_SLOTS
            and aggregate["bytes"] == BANK_BYTES
            and aggregate["failed"] == 0
        ):
            return text, records
        if (
            aggregate["reclaimed"] > BANK_SLOTS
            or aggregate["bytes"] > BANK_BYTES
            or aggregate["failed"] > 0
            or time.monotonic() >= deadline
        ):
            return text, records
        time.sleep(0.05)


def _snapshot_child_state(pool: Any, driver: Any, device: int) -> dict[str, Any]:
    with pool._SLOT_DIRECT_LOCK:
        cache = dict(pool._slot_direct_cache_snapshot_locked())
        policy_plans = len(pool._SLOT_DIRECT_REGISTERED_PLANS)
        policy_generation = pool._SLOT_DIRECT_PLAN_GENERATION
        active_layer = pool._SLOT_DIRECT_PLAN_ACTIVE_LAYER
    with driver._uvm_slot_direct_lock:
        driver_slots = len(driver._uvm_slot_direct_owners.get(device, {}))
        driver_plans = len(driver._uvm_slot_direct_plans.get(device, {}))
        driver_generation = driver._uvm_slot_direct_plan_generations.get(device)
        session_open = device in driver._uvm_slot_direct_sessions
        poison = driver._uvm_slot_direct_poisoned.get(device)
    return {
        "cache": cache,
        "policy_plan_count": policy_plans,
        "policy_generation": policy_generation,
        "policy_active_layer": active_layer,
        "driver_allocated_slots": driver_slots,
        "driver_plan_count": driver_plans,
        "driver_generation": driver_generation,
        "session_open": session_open,
        "poison": poison,
    }


def verify_live_state(
    state: Mapping[str, Any],
    *,
    generation: int,
    ready_slots: int,
) -> None:
    cache = state.get("cache")
    require(isinstance(cache, dict), "state lacks cache")
    require(int(cache.get("allocated_slots", -1)) == BANK_SLOTS, "slot count mismatch")
    require(int(cache.get("ready_slots", -1)) == ready_slots, "ready count mismatch")
    require(int(cache.get("allocated_bytes", -1)) == BANK_BYTES, "bank byte mismatch")
    require(int(state.get("driver_allocated_slots", -1)) == BANK_SLOTS, "driver slot mismatch")
    require(int(state.get("driver_plan_count", -1)) == 5, "driver plan mismatch")
    require(int(state.get("policy_plan_count", -1)) == 4, "policy plan mismatch")
    require(int(state.get("driver_generation", -1)) == generation, "driver generation mismatch")
    require(int(state.get("policy_generation", -1)) == generation, "policy generation mismatch")
    require(state.get("session_open") is True, "slot session is not open")
    require(state.get("poison") is None, "slot session is poisoned")


def verify_mode8_captures(captures: object) -> int:
    require(isinstance(captures, list) and len(captures) == 1, "mode8 call count mismatch")
    result = captures[0]
    require(isinstance(result, dict), "invalid mode8 capture")
    require(result.get("ok") is True and result.get("reason") == "ok", "mode8 failed")
    require(int(result.get("mode", -1)) == 8, "mode8 capture has wrong mode")
    require(int(result.get("ioctl_calls", -1)) == 1, "mode8 logical ioctl mismatch")
    require(int(result.get("requested_count", -1)) == BANK_SLOTS, "mode8 range mismatch")
    require(int(result.get("success_count", -1)) == BANK_SLOTS, "mode8 success mismatch")
    require(int(result.get("requested_bytes", -1)) == BANK_BYTES, "mode8 byte mismatch")
    require(int(result.get("accepted_bytes", -1)) == BANK_BYTES, "mode8 accepted mismatch")
    candidates = int(result.get("fd_candidate_ioctl_calls", -1))
    require(candidates >= 1, "mode8 candidate ioctl count is invalid")
    return candidates


def verify_sigterm_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = payload.get("ready")
    round_marker = payload.get("round")
    child_marker = payload.get("child")
    require(isinstance(ready, dict), "ready marker missing")
    require(isinstance(round_marker, dict), "round marker missing")
    require(isinstance(child_marker, dict), "child marker missing")
    require(ready.get("nonce") == round_marker.get("nonce") == child_marker.get("nonce"), "marker nonce mismatch")
    require(ready.get("pid") == round_marker.get("pid") == child_marker.get("pid"), "marker pid mismatch")
    require(ready.get("sigterm_handler") == "SIG_DFL", "child installed a SIGTERM handler")
    initial_generation = _strict_int(ready.get("generation"), "ready generation", minimum=1)
    verify_live_state(
        ready.get("state", {}),
        generation=initial_generation,
        ready_slots=0,
    )
    candidate_calls = verify_mode8_captures(ready.get("mode8_captures"))
    require(int(round_marker.get("completed_rounds", -1)) == 1, "round count mismatch")
    require(round_marker.get("output_token_ids") == EXPECTED_TOKEN_IDS, "output token mismatch")
    require(round_marker.get("prompt_token_ids_exact") is True, "prompt IDs changed")
    final_generation = initial_generation + LAYERS
    require(int(round_marker.get("generation", -1)) == final_generation, "round generation mismatch")
    verify_live_state(
        round_marker.get("state", {}),
        generation=final_generation,
        ready_slots=BANK_SLOTS,
    )
    require(child_marker.get("status") == "waiting_for_sigterm", "child was not paused")
    require(payload.get("returncode") == -signal.SIGTERM, "child did not exit by SIGTERM")
    require(payload.get("exit_state_exists") is False, "SIGTERM unexpectedly ran atexit observer")
    require(payload.get("escalated_to_sigkill") is False, "controller escalated to SIGKILL")
    signal_ns = _strict_int(payload.get("signal_sent_ns"), "signal time", minimum=1)
    exit_ns = _strict_int(payload.get("process_exit_ns"), "exit time", minimum=signal_ns)
    require(int(round_marker.get("written_epoch_ns", 0)) <= signal_ns <= exit_ns, "signal ordering is invalid")
    delta = payload.get("stats_delta")
    require(isinstance(delta, dict), "stats delta missing")
    expected = expected_sigterm_delta(1, candidate_calls)
    verify_exact_delta(delta, expected)
    route_records = payload.get("route_records")
    require(isinstance(route_records, list), "route records missing")
    reclaim = verify_close_reclaim(route_records)
    gpu = payload.get("gpu_after")
    require(isinstance(gpu, dict), "final GPU snapshot missing")
    require(int(gpu.get("used_mib", -1)) == 0, "GPU memory did not return to 0 MiB")
    require(gpu.get("compute_pids") == [], "GPU still has compute processes")
    return {
        "initial_generation": initial_generation,
        "final_generation": final_generation,
        "candidate_ioctl_calls": candidate_calls,
        "expected_stats": expected,
        "close_reclaim": reclaim,
    }


def verify_recovery_generation(
    payload: Mapping[str, Any], previous_generation: int
) -> int:
    previous_generation = _strict_int(
        previous_generation, "previous generation", minimum=1
    )
    require(payload.get("status") == "pass", "recovery payload failed")
    first_generation = _strict_int(
        payload.get("first_generation"), "recovery first generation", minimum=1
    )
    require(
        first_generation > previous_generation,
        "generation was not monotonic after hard close",
    )
    return first_generation


def _load_common(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("slot_lifecycle_common", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CHILD_LLM_KEEPALIVE: object | None = None


def run_child(args: argparse.Namespace) -> int:
    """Initialize, complete one request, publish the marker, and pause."""
    global _CHILD_LLM_KEEPALIVE
    require(os.environ.get("ALLOW_GPU_RUN") == "1", "child live-run gate is closed")
    require(os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "child must be in-process")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    base = {
        "schema": "vllm_slot_sigterm_child_v1",
        "nonce": args.nonce,
        "pid": pid,
        "status": "starting",
        "started_epoch_ns": time.time_ns(),
    }
    atomic_write_json(work_dir / "child.json", base)

    common = _load_common(args.common_benchmark)
    _metadata, requests = common.load_fixture(args.evaluation_fixture)
    require(0 <= args.fixture_index < len(requests), "fixture index is invalid")
    expected = requests[args.fixture_index]
    require(expected.get("prompt_token_ids") is not None, "fixture lacks frozen token IDs")

    # CUDA-initializing imports are deliberately confined to this code path.
    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.device_allocator import uvm as uvm_driver
    from vllm.device_allocator import uvm_pool_residency as pool

    _target, captures = common.install_slot_preallocate_control_capture(pool)

    def exit_observer(
        pool_module: Any = pool,
        driver_module: Any = uvm_driver,
        output: Path = work_dir / "exit_state.json",
    ) -> None:
        try:
            value = {
                "schema": "vllm_slot_sigterm_exit_observer_v1",
                "nonce": args.nonce,
                "pid": pid,
                "written_epoch_ns": time.time_ns(),
                "state": _snapshot_child_state(pool_module, driver_module, args.device),
            }
        except BaseException as exc:  # atexit must preserve its own evidence
            value = {
                "schema": "vllm_slot_sigterm_exit_observer_v1",
                "nonce": args.nonce,
                "pid": pid,
                "error": repr(exc),
                "written_epoch_ns": time.time_ns(),
            }
        atomic_write_json(output, value)

    # This is intentionally before LLM construction.  Mode 8 registers the
    # UVM cleanup later, so normal atexit would run cleanup before this observer.
    atexit.register(exit_observer)

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.97,
        enforce_eager=True,
        max_model_len=544,
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1_288_490_189,
        cpu_offload_gb=0,
        enable_prefix_caching=False,
        seed=0,
        disable_log_stats=False,
    )
    _CHILD_LLM_KEEPALIVE = llm
    torch.cuda.synchronize()
    handler = signal.getsignal(signal.SIGTERM)
    handler_name = "SIG_DFL" if handler == signal.SIG_DFL else repr(handler)
    state = _snapshot_child_state(pool, uvm_driver, args.device)
    generation = int(state["driver_generation"])
    verify_live_state(state, generation=generation, ready_slots=0)
    verify_mode8_captures(captures)
    ready = {
        "schema": "vllm_slot_sigterm_ready_v1",
        "nonce": args.nonce,
        "pid": pid,
        "written_epoch_ns": time.time_ns(),
        "generation": generation,
        "sigterm_handler": handler_name,
        "state": state,
        "mode8_captures": captures,
    }
    atomic_write_json(work_dir / "ready.json", ready)

    sampling = SamplingParams(
        temperature=0.0,
        seed=0,
        ignore_eos=True,
        min_tokens=1,
        max_tokens=1,
    )
    prompt = TokensPrompt(prompt_token_ids=list(expected["prompt_token_ids"]))
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()
    require(len(outputs) == 1, "generate returned the wrong output count")
    record = common.request_record(outputs[0], expected)
    require(record["prompt_token_ids_exact"] is True, "prompt token IDs changed")
    require(record["output_tokens"] == 1, "completion length changed")
    require(record["output_token_ids"] == EXPECTED_TOKEN_IDS, "completion token changed")
    final_state = _snapshot_child_state(pool, uvm_driver, args.device)
    final_generation = generation + LAYERS
    verify_live_state(
        final_state,
        generation=final_generation,
        ready_slots=BANK_SLOTS,
    )
    round_marker = {
        "schema": "vllm_slot_sigterm_round_v1",
        "nonce": args.nonce,
        "pid": pid,
        "written_epoch_ns": time.time_ns(),
        "completed_rounds": 1,
        "generation": final_generation,
        "prompt_token_ids_exact": record["prompt_token_ids_exact"],
        "output_token_ids": record["output_token_ids"],
        "request": record,
        "state": final_state,
    }
    # Publish the paused state first and the round marker last.  Observing the
    # round marker therefore proves that the child can no longer race ahead.
    atomic_write_json(
        work_dir / "child.json",
        {**base, "status": "waiting_for_sigterm", "round": 1,
         "written_epoch_ns": time.time_ns()},
    )
    atomic_write_json(work_dir / "round.json", round_marker)

    # No timeout and no SIGTERM handler: only the controller may end this child.
    threading.Event().wait()
    raise AssertionError("unreachable")


def build_slot_environment(
    *,
    plan: Path,
    case_dir: Path,
    gpu_index: int,
    cpu_threads: int,
) -> dict[str, str]:
    return {
        "ALLOW_GPU_RUN": "1",
        "VLLM_USE_UVM": "0",
        "VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE": "slot_va_block_remap",
        "VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP": "1",
        "VLLM_UVM_WEIGHT_PREFETCH_ENABLE": "0",
        "VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE": "0",
        "VLLM_UVM_KV_CACHE_MANAGED": "0",
        "VLLM_UVM_KV_RUNTIME_ENABLE": "0",
        "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE": "0",
        "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE": "0",
        "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE": "0",
        "VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE": "0",
        "VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST": "1",
        "VLLM_UVM_WEIGHT_EXECUTOR": "cpp_batch",
        "VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE": "off",
        "VLLM_UVM_MOE_ROUTING_TRACE_ENABLE": "0",
        "VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST": "",
        "VLLM_UVM_EXPERT_ENVELOPE_BYTES": "524288",
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE": "driver_release_cpu",
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE": "post_warmup",
        "VLLM_UVM_INIT_WEIGHT_PLAN_FILE": str(plan),
        "VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN": "1",
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES": str(BANK_BYTES),
        "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS": str(BANK_SLOTS),
        "VLLM_UVM_SLOT_DIRECT_TRACE_FILE": "",
        "VLLM_UVM_SLOT_DIRECT_CONTROL_SUMMARY_FILE": str(case_dir / "control.json"),
        "VLLM_UVM_EXPERT_WEIGHTS_MANAGED": "1",
        "VLLM_UVM_EXPERT_ENVELOPE_ENABLE": "1",
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE": "1",
        "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": "slot_restore",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD": "1",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES": "1",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION": "1",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN": "1",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH": "1",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE": "1",
        "VLLM_CPU_OFFLOAD_BACKEND": "uva",
        "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE": "",
        "VLLM_CPU_OFFLOAD_STATIC_BANK": "0",
        "VLLM_CPU_OFFLOAD_COPY_TRACE": "0",
        "VLLM_CPU_OFFLOAD_COPY_SYNC": "0",
        "VLLM_CPU_OFFLOAD_MANIFEST_FILE": str(case_dir / "cpu_offload_manifest.jsonl"),
        "VLLM_CPU_OFFLOAD_MANIFEST_RESET": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(cpu_threads),
        "MKL_NUM_THREADS": str(cpu_threads),
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
    }


def build_child_command(
    *,
    python: Path,
    script: Path,
    cpuset: str,
    work_dir: Path,
    model: Path,
    fixture: Path,
    nonce: str,
) -> list[str]:
    return [
        "taskset", "-c", cpuset, str(python), str(script), "child",
        "--work-dir", str(work_dir), "--model", str(model),
        "--evaluation-fixture", str(fixture), "--nonce", nonce,
    ]


def _read_int(path: Path, name: str) -> int:
    require(path.is_file(), f"missing {name}: {path}")
    try:
        return int(path.read_text(encoding="utf-8").strip(), 0)
    except ValueError as exc:
        raise LifecycleError(f"invalid {name}: {path}") from exc


def require_live_gate(environment: Mapping[str, str]) -> None:
    require(environment.get("ALLOW_GPU_RUN") == "1", "set ALLOW_GPU_RUN=1")


def _gpu_snapshot(gpu_index: int) -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index),
         "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip().splitlines()
    require(len(query) == 1, "unexpected nvidia-smi GPU output")
    gpu_uuid, used = [part.strip() for part in query[0].split(",")]
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.splitlines()
    pids: list[int] = []
    for line in apps:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == gpu_uuid and parts[1].isdigit():
            pids.append(int(parts[1]))
    return {"gpu_uuid": gpu_uuid, "used_mib": int(used), "compute_pids": pids,
            "epoch_ns": time.time_ns()}


def _wait_gpu_zero(gpu_index: int, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _gpu_snapshot(gpu_index)
        if last["used_mib"] == 0 and not last["compute_pids"]:
            return last
        time.sleep(1.0)
    raise LifecycleError(f"GPU did not become clean: {last}")


def _wait_marker(path: Path, process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return load_json(path)
            except (OSError, json.JSONDecodeError):
                pass
        returncode = process.poll()
        require(returncode is None, f"child exited before {path.name}: {returncode}")
        time.sleep(0.05)
    raise LifecycleError(f"timed out waiting for {path}")


def _write_privileged(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
        return
    except PermissionError:
        pass
    subprocess.run(
        ["sudo", "-n", "tee", str(path)], input=text, text=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True,
    )


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[bool, int, dict[str, Any]]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            reaches_end = not text[index + end:].strip()
            candidates.append((reaches_end, end, value))
    require(bool(candidates), "subprocess did not emit a JSON object")
    # Prefer the outer object which reaches EOF; otherwise use the widest
    # decoded object instead of accidentally selecting its last nested dict.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _runtime_preflight(args: argparse.Namespace) -> str:
    require_live_gate(os.environ)
    for path, label in (
        (VLLM_PY, "vLLM Python"), (COMMON_BENCHMARK, "common benchmark"),
        (PREPARE_INPUTS, "input builder"), (ABORT_PROBE, "abort probe"),
        (DEFAULT_LIBRARY, "UVM shim"), (args.model / "config.json", "model"),
        (args.evaluation_fixture, "fixture"), (args.stats, "latency stats"),
        (args.route_log, "route log"), (args.route_enable, "route enable"),
        (args.srcversion, "srcversion"), (args.module, "custom module"),
    ):
        require(path.is_file(), f"missing {label}: {path}")
    require(shutil.which("nvidia-smi") is not None, "nvidia-smi is unavailable")
    require(shutil.which("taskset") is not None, "taskset is unavailable")
    require(_read_int(args.pte_enable, "PTE remap") == 1, "PTE remap is disabled")
    require(_read_int(args.fast_index, "fast index") == 1, "fast index is disabled")
    require(_read_int(args.direct_cutover, "direct cutover") == 1, "direct cutover is disabled")
    require(_read_int(args.budget, "slot budget") >= BANK_BYTES, "slot budget is too small")
    loaded = args.srcversion.read_text(encoding="utf-8").strip()
    built = subprocess.run(
        ["modinfo", "-F", "srcversion", str(args.module)], check=True,
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    require(loaded == built and bool(loaded), "loaded UVM does not match the module artifact")
    parsed = parse_latency_stats(args.stats.read_text(encoding="utf-8"))
    for name in expected_sigterm_delta(1, 1):
        require(name in parsed, f"loaded UVM lacks telemetry {name}")
    return loaded


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    srcversion = _runtime_preflight(args)
    output = args.output_dir.resolve()
    require(not output.exists(), f"output directory already exists: {output}")
    output.mkdir(parents=True)
    case_dir = output / "sigterm"
    case_dir.mkdir()
    payload: dict[str, Any] = {
        "schema": "vllm_slot_registered_plan_sigterm_v1",
        "status": "failed",
        "output_dir": str(output),
        "srcversion_before": srcversion,
        "exclusive_uvm_workload": True,
    }
    process: subprocess.Popen[Any] | None = None
    route_original: int | None = None
    route_changed = False
    stdout_handle = None
    stderr_handle = None
    lock_handle = (Path("/tmp") / "vllm_slot_registered_plan_lifecycle.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_before = _wait_gpu_zero(args.gpu_index, args.gpu_clean_timeout)
        atomic_write_json(output / "gpu.before.json", gpu_before)
        inputs = output / "inputs"
        with (
            (output / "prepare.stdout").open("w") as prepare_stdout,
            (output / "prepare.stderr").open("w") as prepare_stderr,
        ):
            subprocess.run(
                [str(VLLM_PY), str(PREPARE_INPUTS), "--output-dir", str(inputs)],
                check=True, cwd=REPO_ROOT, stdout=prepare_stdout,
                stderr=prepare_stderr, timeout=60,
            )
        plan = inputs / "exact_l0_l3_plan.json"
        nonce = uuid.uuid4().hex
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith("VLLM_")
        }
        slot_environment = build_slot_environment(
            plan=plan, case_dir=case_dir, gpu_index=args.gpu_index,
            cpu_threads=args.cpu_threads,
        )
        environment.update(slot_environment)
        command = build_child_command(
            python=VLLM_PY, script=Path(__file__).resolve(), cpuset=args.cpuset,
            work_dir=case_dir, model=args.model, fixture=args.evaluation_fixture,
            nonce=nonce,
        )
        # Record only the experiment overlay, never unrelated parent secrets.
        atomic_write_json(
            output / "command.json",
            {"argv": command, "environment": slot_environment},
        )
        before_text = args.stats.read_text(encoding="utf-8")
        (case_dir / "latency_stats.before.txt").write_text(before_text, encoding="utf-8")
        before = parse_latency_stats(before_text)
        stdout_handle = (case_dir / "stdout.log").open("w")
        stderr_handle = (case_dir / "stderr.log").open("w")
        process = subprocess.Popen(
            command, cwd=VLLM_ROOT, env=environment, stdout=stdout_handle,
            stderr=stderr_handle, start_new_session=True,
        )
        ready = _wait_marker(case_dir / "ready.json", process, args.timeout)
        round_marker = _wait_marker(case_dir / "round.json", process, args.timeout)
        child_marker = load_json(case_dir / "child.json")
        require(ready.get("nonce") == nonce, "ready nonce mismatch")
        require(round_marker.get("nonce") == nonce, "round nonce mismatch")
        require(int(ready.get("pid", -1)) == process.pid, "ready PID mismatch")
        require(int(round_marker.get("pid", -1)) == process.pid, "round PID mismatch")
        require(os.getpgid(process.pid) == process.pid, "child lacks an isolated process group")
        require(child_marker.get("status") == "waiting_for_sigterm", "child did not pause")

        route_original = _read_int(args.route_enable, "route-log enable")
        if route_original != 1:
            _write_privileged(args.route_enable, "1\n")
            route_changed = True
        require(_read_int(args.route_enable, "route-log enable") == 1, "could not enable route log")
        route_before = args.route_log.read_text(encoding="utf-8")
        (case_dir / "route.before.txt").write_text(route_before, encoding="utf-8")
        signal_sent_ns = time.time_ns()
        os.killpg(process.pid, signal.SIGTERM)
        escalated = False
        try:
            returncode = process.wait(timeout=args.exit_timeout)
        except subprocess.TimeoutExpired:
            escalated = True
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=30)
        process_exit_ns = time.time_ns()
        # UVM file release can be deferred to a kernel worker after waitpid
        # observes the userspace exit. Keep tracing enabled until that worker
        # has published the complete close-reclaim record.
        route_after, records = _wait_close_reclaim_log(
            args.route_log,
            start_ns=signal_sent_ns,
            timeout=args.exit_timeout,
        )
        (case_dir / "route.after.txt").write_text(route_after, encoding="utf-8")
        if route_changed:
            _write_privileged(args.route_enable, f"{route_original}\n")
            route_changed = False
        stdout_handle.close(); stdout_handle = None
        stderr_handle.close(); stderr_handle = None
        gpu_after = _wait_gpu_zero(args.gpu_index, args.gpu_clean_timeout)
        atomic_write_json(output / "gpu.after_sigterm.json", gpu_after)
        after_text = args.stats.read_text(encoding="utf-8")
        (case_dir / "latency_stats.after.txt").write_text(after_text, encoding="utf-8")
        after = parse_latency_stats(after_text)
        candidate_calls = verify_mode8_captures(ready.get("mode8_captures"))
        expected = expected_sigterm_delta(1, candidate_calls)
        delta = stats_delta(before, after, required=frozenset(expected))
        evidence: dict[str, Any] = {
            "ready": ready, "round": round_marker, "child": child_marker,
            "returncode": returncode, "exit_state_exists": (case_dir / "exit_state.json").exists(),
            "escalated_to_sigkill": escalated, "signal_sent_ns": signal_sent_ns,
            "process_exit_ns": process_exit_ns, "stats_delta": delta,
            "route_records": records, "gpu_after": gpu_after,
        }
        verified = verify_sigterm_evidence(evidence)
        atomic_write_json(case_dir / "evidence.json", evidence)

        recovery_command = [
            "taskset", "-c", args.cpuset, str(VLLM_PY), str(ABORT_PROBE),
            "--library", str(DEFAULT_LIBRARY), "--stats", str(args.stats),
            "--device", "0",
        ]
        recovery = subprocess.run(
            recovery_command, cwd=REPO_ROOT, env=environment, capture_output=True,
            text=True, timeout=args.timeout,
        )
        (output / "recovery.stdout").write_text(recovery.stdout, encoding="utf-8")
        (output / "recovery.stderr").write_text(recovery.stderr, encoding="utf-8")
        require(recovery.returncode == 0, "generation recovery probe failed")
        recovery_json = _extract_json(recovery.stdout)
        verify_recovery_generation(recovery_json, verified["final_generation"])
        gpu_recovery = _wait_gpu_zero(args.gpu_index, args.gpu_clean_timeout)
        require(gpu_recovery["used_mib"] == 0, "recovery left GPU memory")
        final_srcversion = args.srcversion.read_text(encoding="utf-8").strip()
        require(final_srcversion == srcversion, "UVM module changed during the run")
        payload.update({
            "status": "pass", "sigterm": verified,
            "recovery": recovery_json, "gpu_after_recovery": gpu_recovery,
            "srcversion_after": final_srcversion,
        })
    except BaseException as exc:
        payload["error"] = repr(exc)
        payload["error_type"] = type(exc).__name__
        payload["traceback"] = traceback.format_exc()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            except BaseException:
                pass
    finally:
        if route_changed and route_original is not None:
            try:
                _write_privileged(args.route_enable, f"{route_original}\n")
            except BaseException as exc:
                payload["route_restore_error"] = repr(exc)
                payload["status"] = "failed"
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
        lock_handle.close()
        atomic_write_json(output / "summary.json", payload)
    return payload


def _add_path_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--route-log", type=Path, default=DEFAULT_ROUTE_LOG)
    parser.add_argument("--route-enable", type=Path, default=DEFAULT_ROUTE_ENABLE)
    parser.add_argument("--srcversion", type=Path, default=DEFAULT_SRCVERSION)
    parser.add_argument("--pte-enable", type=Path, default=DEFAULT_PTE_ENABLE)
    parser.add_argument("--fast-index", type=Path, default=DEFAULT_FAST_INDEX)
    parser.add_argument("--direct-cutover", type=Path, default=DEFAULT_DIRECT_CUTOVER)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the gated live SIGTERM probe")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    run.add_argument("--evaluation-fixture", type=Path, default=DEFAULT_FIXTURE)
    run.add_argument("--gpu-index", type=int, default=0)
    run.add_argument("--cpuset", default="0-13")
    run.add_argument("--cpu-threads", type=int, default=14)
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument("--exit-timeout", type=float, default=60.0)
    run.add_argument("--gpu-clean-timeout", type=float, default=240.0)
    _add_path_options(run)
    child = subparsers.add_parser("child", help=argparse.SUPPRESS)
    child.add_argument("--work-dir", type=Path, required=True)
    child.add_argument("--model", type=Path, required=True)
    child.add_argument("--evaluation-fixture", type=Path, required=True)
    child.add_argument("--nonce", required=True)
    child.add_argument("--common-benchmark", type=Path, default=COMMON_BENCHMARK)
    child.add_argument("--fixture-index", type=int, default=0)
    child.add_argument("--device", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "child":
        return run_child(args)
    result = run_controller(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
