#!/usr/bin/env python3
"""Validate command-96 fenced registered plans without launching vLLM.

The controller runs every case in an isolated worker.  The positive case
executes a bank-0 bootstrap followed by a bank-1 transition.  A checksum
kernel is ordered before the transition's compute ticket on the same CUDA
stream, and the completion wait is enqueued on that stream only after command
96 reports a committed transition.

The negative cases prove that replayed tickets, non-pool addresses, aliased
compute/completion cells, and legacy command 93 are rejected without owner or
generation publication.  A deliberately invalidated registered allocation is
used to prove the stronger failure rule: the compute fence can be accepted,
while the completion ticket must remain unsignaled when plan execution fails.

This script never loads a module, writes a module parameter, or starts vLLM.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASE_PROBE_PATH = (
    REPO_ROOT / "experiments" / "uvm_slot_registered_plan_probe" / "run_probe.py"
)
DUAL_PROBE_PATH = (
    REPO_ROOT / "experiments" / "uvm_slot_dual_bank_event_probe" / "run_probe.py"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "registered_plan_fence.json"
DEFAULT_FENCE_PARAMETER = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_cuda_fence"
)
DEFAULT_DUAL_PARAMETER = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_dual_bank_event"
)
DEFAULT_COPY_CACHE_PARAMETER = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_cache_copy_runs"
)

EXECUTE_IOCTL = 93
EXECUTE_FENCED_IOCTL = 96
FENCE_FLAGS_ALL = 0x3

CASES = (
    "correct_ticket",
    "replay_ticket",
    "wrong_pool_va",
    "same_address",
    "legacy_ioctl93",
    "failed_execute_no_completion",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_module("fenced_registered_plan_base", BASE_PROBE_PATH)
dual = _load_module("fenced_registered_plan_dual", DUAL_PROBE_PATH)


class ExecuteParams(ctypes.Structure):
    _fields_ = [
        ("plan_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("new_generation", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("conflict_count", ctypes.c_uint32),
        ("remap_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("copy_push_count", ctypes.c_uint32),
        ("owner_cutover", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class ExecuteFencedParams(ctypes.Structure):
    _fields_ = [
        ("plan_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("new_generation", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("conflict_count", ctypes.c_uint32),
        ("remap_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("copy_push_count", ctypes.c_uint32),
        ("owner_cutover", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("fence_flags", ctypes.c_uint32),
        ("compute_semaphore_address", ctypes.c_uint64),
        ("completion_semaphore_address", ctypes.c_uint64),
        ("compute_payload", ctypes.c_uint32),
        ("completion_payload", ctypes.c_uint32),
        ("fence_accepted", ctypes.c_uint32),
        ("completion_signaled", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


def assert_execute_abi() -> None:
    expected = {
        "execute_size": (ctypes.sizeof(ExecuteParams), 88),
        "fenced_size": (ctypes.sizeof(ExecuteFencedParams), 128),
        "fenced_flags": (ExecuteFencedParams.fence_flags.offset, 84),
        "fenced_compute": (
            ExecuteFencedParams.compute_semaphore_address.offset,
            88,
        ),
        "fenced_completion": (
            ExecuteFencedParams.completion_semaphore_address.offset,
            96,
        ),
        "fenced_payload": (ExecuteFencedParams.compute_payload.offset, 104),
        "fenced_accepted": (ExecuteFencedParams.fence_accepted.offset, 112),
        "fenced_signaled": (
            ExecuteFencedParams.completion_signaled.offset,
            116,
        ),
        "fenced_rm_status": (ExecuteFencedParams.rmStatus.offset, 124),
    }
    mismatches = [
        f"{name}={actual}, expected {wanted}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError("command-96 ABI mismatch: " + "; ".join(mismatches))


assert_execute_abi()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise base.ProbeError(message)


def _read_int(path: Path, name: str) -> int:
    require(path.is_file(), f"missing {name}: {path}")
    try:
        return int(path.read_text(encoding="utf-8").strip(), 0)
    except ValueError as exc:
        raise base.ProbeError(f"invalid {name}: {path}") from exc


def _load_cell(address: int) -> int:
    return int(ctypes.c_uint32.from_address(int(address)).value)


def _store_cell(address: int, value: int) -> None:
    ctypes.c_uint32.from_address(int(address)).value = int(value)


def _common_result(
    params: ExecuteParams | ExecuteFencedParams,
    *,
    command: int,
    ret: int,
    errno_value: int,
    wall_ns: int,
) -> dict[str, object]:
    rm_status = int(params.rmStatus)
    return {
        "ioctl_command": int(command),
        "ioctl_ret": int(ret),
        "errno": int(errno_value),
        "rm_status": rm_status,
        "rejected": not (ret == 0 and rm_status == 0),
        "plan_id": int(params.plan_id),
        "layer_id": int(params.layer_id),
        "generation": int(params.generation),
        "new_generation": int(params.new_generation),
        "requested_bytes": int(params.requested_bytes),
        "accepted_bytes": int(params.accepted_bytes),
        "mode": int(params.mode),
        "count": int(params.count),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "conflict_count": int(params.conflict_count),
        "remap_count": int(params.remap_count),
        "tlb_invalidate_count": int(params.tlb_invalidate_count),
        "copy_push_count": int(params.copy_push_count),
        "owner_cutover": int(params.owner_cutover) == 1,
        "poisoned": int(params.poisoned) == 1,
        "ioctl_wall_ns": int(wall_ns),
    }


def _session_resources(backend: Any) -> tuple[int, Any]:
    with backend.uvm._uvm_slot_direct_lock:  # noqa: SLF001 - probe-only raw UAPI
        cached = backend.uvm._uvm_slot_direct_sessions.get(backend.device)  # noqa: SLF001
        require(cached is not None, "registered-plan UVM session disappeared")
        fd, _uuid = cached
        fence = backend.uvm._uvm_slot_direct_cuda_fences.get(backend.device)  # noqa: SLF001
        require(fence is not None, "registered-plan fence pool is missing")
        require(int(fence.fd) == int(fd), "fence pool belongs to another UVM fd")
        return int(fd), fence


def raw_fenced_execute(
    backend: Any,
    *,
    plan_id: int,
    layer_id: int,
    generation: int,
    ticket: int,
    compute_address: int | None = None,
    completion_address: int | None = None,
    signal_compute: bool,
    enqueue_completion_wait: bool,
) -> dict[str, object]:
    fd, fence = _session_resources(backend)
    compute_host = (
        int(fence.compute_address)
        if compute_address is None
        else int(compute_address)
    )
    completion_host = (
        int(fence.completion_address)
        if completion_address is None
        else int(completion_address)
    )
    # Dereference only an aligned cell which is proven to be inside this
    # registered pool. This includes the intentional same-address case (the
    # compute cell), while an arbitrary wrong-pool pointer remains untouched.
    pool_start = int(fence.base)
    pool_end = pool_start + int(fence.length)
    known_completion = (
        completion_host % ctypes.sizeof(ctypes.c_uint32) == 0
        and pool_start <= completion_host
        and completion_host + ctypes.sizeof(ctypes.c_uint32) <= pool_end
    )
    if known_completion:
        _store_cell(completion_host, 0)
    completion_before = _load_cell(completion_host) if known_completion else None

    if signal_compute:
        backend.uvm._uvm_slot_cuda_fence_write_compute(  # noqa: SLF001
            stream=backend.stream,
            address=int(fence.compute_device_address),
            ticket=int(ticket),
        )

    params = ExecuteFencedParams()
    params.plan_id = int(plan_id)
    params.generation = int(generation)
    params.layer_id = int(layer_id)
    params.fence_flags = FENCE_FLAGS_ALL
    params.compute_semaphore_address = compute_host
    params.completion_semaphore_address = completion_host
    params.compute_payload = int(ticket)
    params.completion_payload = int(ticket)
    ctypes.set_errno(0)
    started_ns = time.perf_counter_ns()
    ret = backend.uvm._libc_ioctl(fd, EXECUTE_FENCED_IOCTL, params)  # noqa: SLF001
    wall_ns = time.perf_counter_ns() - started_ns
    errno_value = ctypes.get_errno() if ret != 0 else 0
    result = _common_result(
        params,
        command=EXECUTE_FENCED_IOCTL,
        ret=int(ret),
        errno_value=errno_value,
        wall_ns=wall_ns,
    )
    result.update(
        {
            "ticket": int(ticket),
            "compute_address": compute_host,
            "completion_address": completion_host,
            "compute_signaled_on_stream": bool(signal_compute),
            "fence_accepted": int(params.fence_accepted) == 1,
            "completion_signaled": int(params.completion_signaled) == 1,
            "completion_before": completion_before,
            "completion_after_ioctl": (
                _load_cell(completion_host) if known_completion else None
            ),
            "completion_wait_enqueued": False,
        }
    )

    if enqueue_completion_wait:
        require(not bool(result["rejected"]), "refusing to wait after rejected ioctl")
        require(
            bool(result["completion_signaled"]),
            "refusing to wait for an unsignaled completion",
        )
        require(known_completion, "completion wait does not use the registered cell")
        backend.uvm._uvm_slot_cuda_fence_wait_completion(  # noqa: SLF001
            stream=backend.stream,
            address=int(fence.completion_device_address),
            ticket=int(ticket),
        )
        backend._check_cuda(  # noqa: SLF001 - existing probe adapter
            backend.library.uvm512_stream_synchronize(backend.stream),
            "synchronize command-96 completion wait",
        )
        result["completion_wait_enqueued"] = True
        result["completion_after_wait"] = _load_cell(completion_host)
    else:
        result["completion_after_wait"] = (
            _load_cell(completion_host) if known_completion else None
        )
    return result


def raw_legacy_execute(
    backend: Any, *, plan_id: int, layer_id: int, generation: int
) -> dict[str, object]:
    fd, _fence = _session_resources(backend)
    params = ExecuteParams()
    params.plan_id = int(plan_id)
    params.generation = int(generation)
    params.layer_id = int(layer_id)
    ctypes.set_errno(0)
    started_ns = time.perf_counter_ns()
    ret = backend.uvm._libc_ioctl(fd, EXECUTE_IOCTL, params)  # noqa: SLF001
    wall_ns = time.perf_counter_ns() - started_ns
    errno_value = ctypes.get_errno() if ret != 0 else 0
    return _common_result(
        params,
        command=EXECUTE_IOCTL,
        ret=int(ret),
        errno_value=errno_value,
        wall_ns=wall_ns,
    )


def _require_no_publication(
    response: Mapping[str, object], *, generation: int, stage: str
) -> None:
    require(bool(response.get("rejected")), f"{stage}: ioctl unexpectedly succeeded")
    require(
        response.get("owner_cutover") is False,
        f"{stage}: owner was published on rejection",
    )
    require(
        int(response.get("accepted_bytes", 0)) == 0,
        f"{stage}: bytes were published on rejection",
    )
    require(
        int(response.get("success_count", 0)) == 0,
        f"{stage}: success count was published on rejection",
    )
    require(
        int(response.get("remap_count", 0)) == 0,
        f"{stage}: remap count was published on rejection",
    )
    require(
        int(response.get("new_generation", -1)) == int(generation),
        f"{stage}: active generation changed on rejection",
    )


def _require_fenced_success(
    response: Mapping[str, object],
    *,
    generation: int,
    ticket: int,
    owner_cutover: bool,
    stage: str,
) -> None:
    require(
        int(response.get("ioctl_command", -1)) == EXECUTE_FENCED_IOCTL,
        f"{stage}: wrong ioctl command",
    )
    require(response.get("rejected") is False, f"{stage}: command 96 failed")
    require(response.get("fence_accepted") is True, f"{stage}: fence not accepted")
    require(
        response.get("completion_signaled") is True,
        f"{stage}: completion not signaled",
    )
    require(
        int(response.get("new_generation", -1)) == generation + 1,
        f"{stage}: generation did not advance exactly once",
    )
    require(
        response.get("owner_cutover") is owner_cutover,
        f"{stage}: owner-cutover result mismatch",
    )
    require(
        int(response.get("completion_after_ioctl", -1)) == ticket,
        f"{stage}: CPU completion cell does not contain the ticket",
    )
    require(
        response.get("completion_wait_enqueued") is True,
        f"{stage}: stream completion wait was not enqueued",
    )
    require(
        int(response.get("completion_after_wait", -1)) == ticket,
        f"{stage}: completion changed across the stream wait",
    )


def require_case_contract(case: str, payload: Mapping[str, object]) -> None:
    require(case in CASES, f"unknown fence case {case}")
    require(payload.get("case") == case, f"{case}: artifact case mismatch")
    generation = int(payload.get("generation_start", 0))
    require(generation > 0, f"{case}: invalid starting generation")
    responses = payload.get("responses")
    require(isinstance(responses, list), f"{case}: responses are missing")

    if case == "correct_ticket":
        require(len(responses) == 2, "correct_ticket: expected bootstrap + transition")
        first, second = responses
        require(isinstance(first, Mapping) and isinstance(second, Mapping), "bad response")
        _require_fenced_success(
            first,
            generation=generation,
            ticket=1,
            owner_cutover=False,
            stage="correct_ticket bootstrap",
        )
        _require_fenced_success(
            second,
            generation=generation + 1,
            ticket=2,
            owner_cutover=True,
            stage="correct_ticket transition",
        )
        require(
            int(payload.get("generation_end", -1)) == generation + 2,
            "correct_ticket: final generation mismatch",
        )
        require(
            payload.get("compute_checksum_completed_before_ticket") is True,
            "correct_ticket: compute ordering was not proven",
        )
        return

    if case == "replay_ticket":
        require(len(responses) == 2, "replay_ticket: expected success + replay")
        accepted, rejected = responses
        require(
            isinstance(accepted, Mapping) and isinstance(rejected, Mapping),
            "bad replay response",
        )
        _require_fenced_success(
            accepted,
            generation=generation,
            ticket=1,
            owner_cutover=False,
            stage="replay_ticket bootstrap",
        )
        _require_no_publication(
            rejected,
            generation=generation + 1,
            stage="replay_ticket rejection",
        )
        require(
            rejected.get("fence_accepted") is False,
            "replay_ticket: stale ticket was accepted",
        )
        require(
            rejected.get("completion_signaled") is False,
            "replay_ticket: rejection signaled completion",
        )
        require(
            int(rejected.get("completion_after_ioctl", -1)) == 0,
            "replay_ticket: completion cell changed",
        )
        return

    require(len(responses) == 1, f"{case}: expected one response")
    response = responses[0]
    require(isinstance(response, Mapping), f"{case}: malformed response")
    _require_no_publication(response, generation=generation, stage=case)

    if case == "legacy_ioctl93":
        require(
            int(response.get("ioctl_command", -1)) == EXECUTE_IOCTL,
            "legacy_ioctl93: command 93 was not exercised",
        )
        return

    require(
        int(response.get("ioctl_command", -1)) == EXECUTE_FENCED_IOCTL,
        f"{case}: command 96 was not exercised",
    )
    require(
        response.get("completion_signaled") is False,
        f"{case}: failed command signaled completion",
    )
    require(
        int(response.get("completion_before", -1)) == 0,
        f"{case}: completion cell was not reset before ioctl",
    )
    require(
        int(response.get("completion_after_ioctl", -1)) == 0,
        f"{case}: failed command changed completion cell",
    )
    if case in ("wrong_pool_va", "same_address"):
        require(
            response.get("fence_accepted") is False,
            f"{case}: invalid address contract was accepted",
        )
    elif case == "failed_execute_no_completion":
        require(
            response.get("fence_accepted") is True,
            "failed_execute_no_completion: compute fence was not accepted first",
        )
    else:  # pragma: no cover - CASES and branches are deliberately exhaustive
        raise base.ProbeError(f"unhandled fence case {case}")


def _setup_session(backend: Any) -> dict[str, object]:
    blocks = 1
    datasets: dict[int, int] = {}
    pointers: dict[int, tuple[int, ...]] = {}
    for layer_id, seed in ((0, base.LAYER_SEEDS[0]), (1, base.LAYER_SEEDS[1])):
        dataset = int(backend.create_dataset(layer_id, seed, blocks))
        datasets[layer_id] = dataset
        pointers[layer_id] = backend.dataset_pointers(dataset, blocks)

    bank0 = dual.build_bank_entries(
        pointers[0], logical_layer=0, bank_id=0, reuse_required=False
    )
    bank1 = dual.build_bank_entries(
        pointers[1], logical_layer=1, bank_id=1, reuse_required=False
    )
    preallocate0 = backend.preallocate(bank0)
    preallocate1 = backend.preallocate(bank1)
    base.require_preallocate(
        preallocate0, entries=base.SLICES_PER_BLOCK, logical_bytes=base.BLOCK_BYTES
    )
    base.require_preallocate(
        preallocate1, entries=base.SLICES_PER_BLOCK, logical_bytes=base.BLOCK_BYTES
    )

    initial = backend.register_plan(
        base.INITIAL_PLAN_ID, 0, bank0, transition=False
    )
    base.require_register(
        initial,
        plan_id=base.INITIAL_PLAN_ID,
        layer_id=0,
        mode=base.MODE_INITIAL,
        entries=base.SLICES_PER_BLOCK,
        logical_bytes=base.BLOCK_BYTES,
    )
    generation = int(initial["generation"])
    transition_entries = dual.build_bank_entries(
        pointers[1], logical_layer=1, bank_id=1, reuse_required=True
    )
    transition_plan = base.transition_plan_id(1)
    transition = backend.register_plan(
        transition_plan, 1, transition_entries, transition=True
    )
    base.require_register(
        transition,
        plan_id=transition_plan,
        layer_id=1,
        mode=base.MODE_TRANSITION,
        entries=base.SLICES_PER_BLOCK,
        logical_bytes=base.BLOCK_BYTES,
        generation=generation,
    )
    _fd, fence = _session_resources(backend)
    return {
        "datasets": datasets,
        "generation": generation,
        "transition_plan": transition_plan,
        "fence_base": int(fence.base),
        "fence_length": int(fence.length),
        "preallocate": [base.summarize(preallocate0), base.summarize(preallocate1)],
        "registrations": [base.summarize(initial), base.summarize(transition)],
    }


def _track_raw_generation(backend: Any, generation: int) -> None:
    # Raw command-96 calls intentionally bypass the production wrapper.  Keep
    # only its cleanup generation mirror in sync; close() still performs the
    # generation-free session-abort audit after destroying the plans.
    with backend.uvm._uvm_slot_direct_lock:  # noqa: SLF001
        backend.uvm._uvm_slot_direct_plan_generations[backend.device] = int(  # noqa: SLF001
            generation
        )


def _launch_checksum_without_host_wait(backend: Any, dataset: int) -> None:
    backend._check_cuda(  # noqa: SLF001
        backend.library.uvm512_dataset_launch_checksum(dataset, backend.stream),
        "launch pre-fence checksum",
    )


def _read_checksum(backend: Any, dataset: int) -> tuple[int, int]:
    actual = ctypes.c_uint64()
    backend._check_cuda(  # noqa: SLF001
        backend.library.uvm512_dataset_read_checksum(dataset, ctypes.byref(actual)),
        "read pre-fence checksum",
    )
    expected = int(backend.library.uvm512_dataset_expected_checksum(dataset))
    return int(actual.value), expected


def _create_torch_cuda_anchor(
    device: int,
) -> tuple[Any, Any, dict[str, object]]:
    """Initialize PyTorch's primary CUDA context before creating the UVM session.

    The CUDA shim used by ``GpuBackend`` can create datasets without making
    PyTorch establish its own primary-context/allocation state.  Real vLLM has
    already done that before slot-plan registration.  Keeping one ordinary
    PyTorch CUDA allocation alive also makes the probe's cuMemHostRegister path
    match the serving process rather than a shim-only CUDA process.
    """
    # Import and initialize PyTorch before GpuBackend loads the CUDA shim. The
    # shim creates its own stream in __init__, which is already too late for
    # reproducing vLLM's primary-context/file-description topology.
    import torch

    device = int(device)
    with torch.cuda.device(device):
        anchor = torch.empty(
            (1,),
            dtype=torch.uint8,
            device=torch.device("cuda", device),
        )
        # Force PyTorch to materialize both the allocation and current-stream
        # objects before preallocate chooses/adopts the registered-plan UVM fd.
        address = int(anchor.data_ptr())
        stream = int(torch.cuda.current_stream(device).cuda_stream)
    require(address != 0, "PyTorch CUDA anchor has a null device address")
    return anchor, torch, {
        "bytes": int(anchor.numel() * anchor.element_size()),
        "device": device,
        "device_address": address,
        "stream": stream,
    }


def run_case(args: argparse.Namespace, case: str) -> dict[str, object]:
    require(case in CASES, f"unknown case {case}")
    os.environ["VLLM_UVM_SLOT_DIRECT_CUDA_FENCE"] = "1"
    # Real vLLM has materialized ordinary PyTorch CUDA allocations before it
    # initializes the registered slot bank. Do this before GpuBackend invokes
    # any cudart/shim operation, then retain the allocation through teardown.
    torch_anchor, torch_module, anchor_info = _create_torch_cuda_anchor(
        args.device
    )
    backend = base.GpuBackend(
        library=args.library,
        stats=args.stats,
        device=args.device,
    )
    payload: dict[str, object] = {
        "status": "failed",
        "case": case,
        "torch_cuda_anchor": anchor_info,
    }
    try:
        setup = _setup_session(backend)
        datasets = setup["datasets"]
        require(isinstance(datasets, dict), "dataset setup is malformed")
        generation = int(setup["generation"])
        transition_plan = int(setup["transition_plan"])
        responses: list[dict[str, object]] = []
        payload.update(
            {
                "generation_start": generation,
                "fence_base": setup["fence_base"],
                "fence_length": setup["fence_length"],
                "preallocate": setup["preallocate"],
                "registrations": setup["registrations"],
                "compute_copy_overlap_allowed": False,
            }
        )

        if case in ("correct_ticket", "replay_ticket"):
            first = raw_fenced_execute(
                backend,
                plan_id=base.INITIAL_PLAN_ID,
                layer_id=0,
                generation=generation,
                ticket=1,
                signal_compute=True,
                enqueue_completion_wait=True,
            )
            responses.append(first)
            require(not bool(first["rejected"]), "bootstrap command 96 failed")
            generation = int(first["new_generation"])
            _track_raw_generation(backend, generation)

            if case == "correct_ticket":
                _launch_checksum_without_host_wait(backend, int(datasets[0]))
                second = raw_fenced_execute(
                    backend,
                    plan_id=transition_plan,
                    layer_id=1,
                    generation=generation,
                    ticket=2,
                    signal_compute=True,
                    enqueue_completion_wait=True,
                )
                responses.append(second)
                require(not bool(second["rejected"]), "transition command 96 failed")
                generation = int(second["new_generation"])
                _track_raw_generation(backend, generation)
                actual, expected = _read_checksum(backend, int(datasets[0]))
                require(actual == expected, "pre-fence checksum mismatch")
                target_actual, target_expected = backend.checksum(int(datasets[1]))
                require(target_actual == target_expected, "new-owner checksum mismatch")
                payload.update(
                    {
                        "compute_checksum": actual,
                        "compute_checksum_expected": expected,
                        "target_checksum": target_actual,
                        "target_checksum_expected": target_expected,
                        "compute_checksum_completed_before_ticket": True,
                    }
                )
            else:
                replay = raw_fenced_execute(
                    backend,
                    plan_id=transition_plan,
                    layer_id=1,
                    generation=generation,
                    ticket=1,
                    signal_compute=False,
                    enqueue_completion_wait=False,
                )
                responses.append(replay)

        elif case == "wrong_pool_va":
            wrong_cell = ctypes.c_uint32(0)
            wrong_address = ctypes.addressof(wrong_cell)
            fence_base = int(setup["fence_base"])
            fence_end = fence_base + int(setup["fence_length"])
            require(
                not fence_base <= wrong_address < fence_end,
                "ctypes wrong-pool cell unexpectedly overlaps the UVM pool",
            )
            responses.append(
                raw_fenced_execute(
                    backend,
                    plan_id=base.INITIAL_PLAN_ID,
                    layer_id=0,
                    generation=generation,
                    ticket=1,
                    compute_address=wrong_address,
                    signal_compute=False,
                    enqueue_completion_wait=False,
                )
            )
            payload["wrong_pool_address"] = wrong_address

        elif case == "same_address":
            _fd, fence = _session_resources(backend)
            responses.append(
                raw_fenced_execute(
                    backend,
                    plan_id=base.INITIAL_PLAN_ID,
                    layer_id=0,
                    generation=generation,
                    ticket=1,
                    completion_address=int(fence.compute_address),
                    signal_compute=False,
                    enqueue_completion_wait=False,
                )
            )
        elif case == "legacy_ioctl93":
            responses.append(
                raw_legacy_execute(
                    backend,
                    plan_id=base.INITIAL_PLAN_ID,
                    layer_id=0,
                    generation=generation,
                )
            )

        elif case == "failed_execute_no_completion":
            backend.destroy_dataset(int(datasets[0]))
            response = raw_fenced_execute(
                backend,
                plan_id=base.INITIAL_PLAN_ID,
                layer_id=0,
                generation=generation,
                ticket=1,
                signal_compute=True,
                enqueue_completion_wait=False,
            )
            responses.append(response)
            payload["registered_allocation_invalidated"] = True

        payload["responses"] = responses
        payload["generation_end"] = generation
        require_case_contract(case, payload)
        payload["status"] = "pass"
    finally:
        cleanup = backend.finish()
        payload["cleanup"] = cleanup
        if not bool(cleanup.get("ok")):
            payload["status"] = "failed"
            payload["cleanup_error"] = cleanup.get("errors")
        # Release only after fence-pool/session cleanup so the primary CUDA
        # context and ordinary CUDA allocation stay live through
        # cuMemHostUnregister, UVM_FREE, and fd close.
        if torch_anchor is not None:
            torch_anchor = None
            torch_module.cuda.empty_cache()
            payload["torch_cuda_anchor_released_after_cleanup"] = True
    require(payload.get("status") == "pass", f"{case}: cleanup failed")
    return payload


def _gpu_used_mib(device: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(device),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def _wait_for_zero_mib(device: int, timeout_seconds: float = 15.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        used = _gpu_used_mib(device)
        if used == 0 or time.monotonic() >= deadline:
            return used
        time.sleep(0.25)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _worker_command(args: argparse.Namespace, case: str, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        case,
        "--worker-output",
        str(output),
        "--library",
        str(args.library),
        "--stats",
        str(args.stats),
        "--fence-parameter",
        str(args.fence_parameter),
        "--dual-parameter",
        str(args.dual_parameter),
        "--copy-cache-parameter",
        str(args.copy_cache_parameter),
        "--device",
        str(args.device),
    ]


def run_worker(args: argparse.Namespace) -> int:
    case = str(args.worker_case)
    payload: dict[str, object] = {"status": "failed", "case": case}
    try:
        require(_gpu_used_mib(args.device) == 0, "GPU is not idle before worker")
        require(
            _read_int(args.fence_parameter, "CUDA-fence parameter") == 1,
            "uvm_moe_slot_plan_cuda_fence must be enabled",
        )
        require(
            _read_int(args.dual_parameter, "dual-bank parameter") == 1,
            "uvm_moe_slot_plan_dual_bank_event must be enabled",
        )
        require(
            _read_int(args.copy_cache_parameter, "copy-cache parameter") == 1,
            "uvm_moe_slot_plan_cache_copy_runs must be enabled",
        )
        payload = run_case(args, case)
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    require(args.worker_output is not None, "worker output is missing")
    _write_json(args.worker_output, payload)
    return 0 if payload.get("status") == "pass" else 1


def run_controller(args: argparse.Namespace) -> int:
    selected = CASES if args.case == "all" else (str(args.case),)
    payload: dict[str, object] = {
        "status": "failed",
        "schema": "uvm_registered_plan_cuda_fence_v1",
        "cases_requested": list(selected),
    }
    try:
        require(_gpu_used_mib(args.device) == 0, "GPU is not idle before probe")
        results: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="uvm-fenced-plan-") as temp:
            temp_path = Path(temp)
            for case in selected:
                worker_output = temp_path / f"{case}.json"
                completed = subprocess.run(
                    _worker_command(args, case, worker_output),
                    text=True,
                    capture_output=True,
                    timeout=180,
                    check=False,
                )
                require(worker_output.is_file(), f"{case}: worker produced no artifact")
                result = json.loads(worker_output.read_text(encoding="utf-8"))
                result["worker_returncode"] = completed.returncode
                result["worker_stdout"] = completed.stdout
                result["worker_stderr"] = completed.stderr
                results.append(result)
                require(completed.returncode == 0, f"{case}: worker failed")
                require(result.get("status") == "pass", f"{case}: artifact failed")
                require(_wait_for_zero_mib(args.device) == 0, f"{case}: GPU leak")
        payload.update(
            {
                "status": "pass",
                "results": results,
                "cases_passed": [result["case"] for result in results],
                "gpu_memory_final_mib": _gpu_used_mib(args.device),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "gpu_memory_final_mib": _wait_for_zero_mib(args.device),
            }
        )
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "pass" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all",) + CASES, default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--library", type=Path, default=base.DEFAULT_LIBRARY)
    parser.add_argument("--stats", type=Path, default=base.DEFAULT_STATS)
    parser.add_argument(
        "--fence-parameter", type=Path, default=DEFAULT_FENCE_PARAMETER
    )
    parser.add_argument("--dual-parameter", type=Path, default=DEFAULT_DUAL_PARAMETER)
    parser.add_argument(
        "--copy-cache-parameter", type=Path, default=DEFAULT_COPY_CACHE_PARAMETER
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be non-negative")
    if (args.worker_case is None) != (args.worker_output is None):
        parser.error("worker case and worker output must be provided together")
    return args


def main() -> int:
    args = parse_args()
    if args.worker_case is not None:
        return run_worker(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
