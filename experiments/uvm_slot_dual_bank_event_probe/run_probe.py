#!/usr/bin/env python3
"""Validate event-guarded dual-bank registered-plan transitions without vLLM.

The probe intentionally keeps compute and transfer serial. A checksum kernel is
launched on one CUDA stream, a CUDA event is recorded on that stream, and the
host waits for the event before issuing the next compact transition ioctl. The
kernel may overlap only the inactive-bank CE copy with old-PTE/target-PTE work.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASE_PROBE_PATH = REPO_ROOT / "experiments" / "uvm_slot_registered_plan_probe" / "run_probe.py"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "dual_bank_event.json"
DEFAULT_DUAL_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_dual_bank_event"
)
DEFAULT_CACHE_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_cache_copy_runs"
)
DEFAULT_PREWARM_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_register_pte_prewarm"
)
DEFAULT_OVERLAP_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_overlap_copy_prepare"
)
DEFAULT_FAST_VALIDATION_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_fast_validation"
)
DEFAULT_SHADOW_PTE_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_shadow_pte_prefill"
)


def load_base_probe():
    spec = importlib.util.spec_from_file_location("registered_plan_probe", BASE_PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_probe()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_int(path: Path, name: str) -> int:
    require(path.is_file(), f"missing {name}: {path}")
    return int(path.read_text(encoding="utf-8").strip(), 0)


def build_bank_entries(
    pointers: Sequence[int],
    *,
    logical_layer: int,
    bank_id: int,
    reuse_required: bool,
) -> tuple[tuple[int, ...], ...]:
    require(bank_id in (0, 1), "bank_id must be 0 or 1")
    flags = base.NO_EVICT | (
        base.REUSE_BACKING_REQUIRED if reuse_required else 0
    )
    entries: list[tuple[int, ...]] = []
    driver_layer = base.DRIVER_BANK_LAYER + bank_id
    owner_base = 0x10000 * (logical_layer + 1)
    for block_index, pointer in enumerate(pointers):
        for slice_index in range(base.SLICES_PER_BLOCK):
            entry_index = block_index * base.SLICES_PER_BLOCK + slice_index
            entries.append(
                (
                    int(pointer) + slice_index * base.SLICE_BYTES,
                    base.SLICE_BYTES,
                    driver_layer,
                    owner_base + entry_index,
                    entry_index + 1,
                    0,
                    flags,
                )
            )
    return tuple(entries)


def cuda_compute_event_barrier(
    backend: Any, event: int, dataset: int
) -> dict[str, int]:
    start_ns = time.perf_counter_ns()
    backend._check_cuda(  # noqa: SLF001 - experimental probe
        backend.library.uvm512_dataset_launch_checksum(dataset, backend.stream),
        "launch event-guarded checksum",
    )
    record_ns = time.perf_counter_ns()
    backend._check_cuda(  # noqa: SLF001
        backend.library.uvm512_event_record(event, backend.stream),
        "record compute-done event",
    )
    wait_start_ns = time.perf_counter_ns()
    backend._check_cuda(  # noqa: SLF001
        backend.library.uvm512_event_synchronize(event),
        "wait compute-done event",
    )
    wait_end_ns = time.perf_counter_ns()

    actual = base.ctypes.c_uint64()
    backend._check_cuda(  # noqa: SLF001
        backend.library.uvm512_dataset_read_checksum(dataset, base.ctypes.byref(actual)),
        "read event-guarded checksum",
    )
    expected = int(backend.library.uvm512_dataset_expected_checksum(dataset))
    require(int(actual.value) == expected, "event-guarded checksum mismatch")
    return {
        "launch_to_record_ns": record_ns - start_ns,
        "event_wait_ns": wait_end_ns - wait_start_ns,
        "barrier_wall_ns": wait_end_ns - start_ns,
        "completed_ns": wait_end_ns,
        "checksum": int(actual.value),
        "expected": expected,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    blocks = int(args.blocks)
    entries_per_bank = blocks * base.SLICES_PER_BLOCK
    bytes_per_bank = blocks * base.BLOCK_BYTES
    parameters = {
        "dual_bank_event": read_int(args.dual_parameter, "dual-bank parameter"),
        "copy_run_cache": read_int(args.cache_parameter, "copy-cache parameter"),
        "pte_prewarm": read_int(args.prewarm_parameter, "PTE-prewarm parameter"),
        "legacy_overlap": read_int(args.overlap_parameter, "overlap parameter"),
        "fast_validation": read_int(
            args.fast_validation_parameter, "fast-validation parameter"
        ),
        "shadow_pte_prefill": read_int(
            args.shadow_pte_parameter, "shadow-PTE parameter"
        ),
        "fast_index": read_int(args.fast_index_parameter, "fast-index parameter"),
        "direct_cutover": read_int(
            args.direct_cutover_parameter, "direct-cutover parameter"
        ),
        "vidmem_budget_bytes": read_int(args.budget_parameter, "VIDMEM budget"),
    }
    require(parameters["dual_bank_event"] == 1, "dual-bank event mode must be enabled")
    require(parameters["copy_run_cache"] == 1, "copy-run cache must be enabled")
    require(parameters["pte_prewarm"] == 1, "PTE prewarm must be enabled")
    require(parameters["legacy_overlap"] == 0, "legacy overlap must be disabled")
    require(parameters["fast_index"] == 1, "fast index must be enabled")
    require(parameters["direct_cutover"] == 1, "direct cutover must be enabled")
    require(
        parameters["vidmem_budget_bytes"] >= 2 * bytes_per_bank,
        "VIDMEM budget is smaller than two banks",
    )

    backend = base.GpuBackend(
        library=args.library,
        stats=args.stats,
        device=args.device,
    )
    event = 0
    datasets: dict[int, int] = {}
    try:
        event = int(backend.library.uvm512_event_create())
        require(event != 0, "cudaEventCreate failed")

        pointers: dict[int, tuple[int, ...]] = {}
        for layer_id, seed in zip(base.LAYER_IDS, base.LAYER_SEEDS):
            dataset = int(backend.create_dataset(layer_id, seed, blocks))
            datasets[layer_id] = dataset
            pointers[layer_id] = backend.dataset_pointers(dataset, blocks)

        bank0_prealloc = build_bank_entries(
            pointers[0], logical_layer=0, bank_id=0, reuse_required=False
        )
        bank1_prealloc = build_bank_entries(
            pointers[1], logical_layer=1, bank_id=1, reuse_required=False
        )
        before_preallocate = backend.read_stats()
        preallocate0 = backend.preallocate(bank0_prealloc)
        preallocate1 = backend.preallocate(bank1_prealloc)
        after_preallocate = backend.read_stats()
        base.require_preallocate(
            preallocate0, entries=entries_per_bank, logical_bytes=bytes_per_bank
        )
        base.require_preallocate(
            preallocate1, entries=entries_per_bank, logical_bytes=bytes_per_bank
        )

        transition_entries = {
            layer_id: build_bank_entries(
                pointers[layer_id],
                logical_layer=layer_id,
                bank_id=layer_id & 1,
                reuse_required=True,
            )
            for layer_id in base.LAYER_IDS
        }

        registrations: list[dict[str, object]] = []
        initial = backend.register_plan(
            base.INITIAL_PLAN_ID,
            0,
            bank0_prealloc,
            transition=False,
        )
        base.require_register(
            initial,
            plan_id=base.INITIAL_PLAN_ID,
            layer_id=0,
            mode=base.MODE_INITIAL,
            entries=entries_per_bank,
            logical_bytes=bytes_per_bank,
        )
        generation = int(initial["generation"])
        registrations.append(base.summarize(initial))
        for layer_id in base.LAYER_IDS:
            plan_id = base.transition_plan_id(layer_id)
            result = backend.register_plan(
                plan_id,
                layer_id,
                transition_entries[layer_id],
                transition=True,
            )
            base.require_register(
                result,
                plan_id=plan_id,
                layer_id=layer_id,
                mode=base.MODE_TRANSITION,
                entries=entries_per_bank,
                logical_bytes=bytes_per_bank,
                generation=generation,
            )
            registrations.append(base.summarize(result))

        execution_order = (0, 1, 2, 3, 0)
        executions: list[dict[str, object]] = []
        profile_deltas: list[dict[str, int]] = []
        event_barriers: list[dict[str, int]] = []
        active_bank: int | None = None
        event_completed_ns: int | None = None

        for sequence_index, layer_id in enumerate(execution_order):
            initial_execute = sequence_index == 0
            target_bank = layer_id & 1
            if not initial_execute:
                require(active_bank is not None, "transition has no active bank")
                require(target_bank != active_bank, "transition did not alternate banks")
                require(event_completed_ns is not None, "missing compute-done event")

            plan_id = (
                base.INITIAL_PLAN_ID
                if initial_execute
                else base.transition_plan_id(layer_id)
            )
            mode = base.MODE_INITIAL if initial_execute else base.MODE_TRANSITION
            before = backend.read_stats()
            execute_start_ns = time.perf_counter_ns()
            if event_completed_ns is not None:
                require(
                    execute_start_ns >= event_completed_ns,
                    "transition began before the compute event completed",
                )
            result = backend.execute_plan(plan_id, layer_id, generation)
            execute_end_ns = time.perf_counter_ns()
            after = backend.read_stats()
            delta = base.stats_delta(before, after)
            base.require_execute(
                result,
                plan_id=plan_id,
                layer_id=layer_id,
                generation=generation,
                mode=mode,
                entries=entries_per_bank,
                blocks=blocks,
                logical_bytes=bytes_per_bank,
            )
            expected_dual_calls = 0 if initial_execute else 1
            require(
                int(delta.get("slot_direct_dual_bank_copy_first_calls", -1))
                == expected_dual_calls,
                "dual-bank copy-first telemetry mismatch",
            )
            generation += 1
            active_bank = target_bank

            barrier = cuda_compute_event_barrier(backend, event, datasets[layer_id])
            barrier["sequence_index"] = sequence_index
            barrier["layer_id"] = layer_id
            barrier["bank_id"] = target_bank
            barrier["execute_end_ns"] = execute_end_ns
            event_completed_ns = int(barrier["completed_ns"])
            event_barriers.append(barrier)
            executions.append(
                {
                    **base.summarize(result),
                    "bank_id": target_bank,
                    "execute_outer_ns": execute_end_ns - execute_start_ns,
                    "compute_event_completed_before_execute": (
                        True if initial_execute else execute_start_ns >= int(
                            event_barriers[-2]["completed_ns"]
                        )
                    ),
                }
            )
            profile_deltas.append(delta)

        destroy_results = []
        for plan_id in (base.INITIAL_PLAN_ID,) + tuple(
            base.transition_plan_id(layer) for layer in base.LAYER_IDS
        ):
            result = backend.destroy_plan(plan_id, generation)
            require(bool(result.get("ok")), f"destroy plan {plan_id} failed")
            destroy_results.append(base.summarize(result))

        pre_close_state = backend.state_snapshot()
        require(
            int(pre_close_state.get("allocated_slots", -1)) == 2 * entries_per_bank,
            "two-bank slot count mismatch before close",
        )
        require(
            int(pre_close_state.get("plan_count", -1)) == 0,
            "userspace still tracks a destroyed plan",
        )
        close = backend.close()
        require(bool(close.get("ok")), f"close failed: {close}")
        require(close.get("closed") is True, "slot session was not closed")
        release = close.get("release")
        require(isinstance(release, dict), "close has no release result")
        require(bool(release.get("ok")), f"slot release failed: {release}")
        require(
            int(release.get("requested_count", -1)) == 2 * entries_per_bank,
            "close requested the wrong two-bank slot count",
        )
        require(
            int(release.get("success_count", -1)) == 2 * entries_per_bank,
            "close did not release both banks",
        )
        require(int(release.get("failure_count", -1)) == 0, "close release failed")
        require(
            int(release.get("accepted_bytes", -1)) == 2 * bytes_per_bank,
            "close released the wrong two-bank byte count",
        )
        post_close_state = backend.state_snapshot()
        expected_zero = {
            "allocated_slots": 0,
            "plan_count": 0,
            "tracked_generation": None,
            "session_open": False,
        }
        for name, value in expected_zero.items():
            require(
                post_close_state.get(name) == value,
                f"post-close {name}={post_close_state.get(name)!r}, expected {value!r}",
            )

        return {
            "status": "pass",
            "blocks": blocks,
            "entries_per_bank": entries_per_bank,
            "bytes_per_bank": bytes_per_bank,
            "total_bank_bytes": 2 * bytes_per_bank,
            "module_parameters": parameters,
            "module_srcversion": args.srcversion.read_text(encoding="utf-8").strip(),
            "execution_order": list(execution_order),
            "bank_order": [layer & 1 for layer in execution_order],
            "compute_copy_overlap_allowed": False,
            "compute_event_waited_before_each_transition": True,
            "preallocate": [base.summarize(preallocate0), base.summarize(preallocate1)],
            "preallocate_profile_delta": base.stats_delta(
                before_preallocate, after_preallocate
            ),
            "registrations": registrations,
            "executions": executions,
            "execute_profile_deltas": profile_deltas,
            "event_barriers": event_barriers,
            "destroy_results": destroy_results,
            "pre_close_state": pre_close_state,
            "close": base.summarize(close),
            "post_close_state": post_close_state,
        }
    finally:
        if event:
            backend.library.uvm512_event_destroy(event)
        cleanup = backend.finish()
        require(bool(cleanup.get("ok")), f"backend cleanup failed: {cleanup}")


def gpu_used_mib(device: int) -> int:
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


def wait_for_zero_mib(
    device: int, timeout_seconds: float = 15.0
) -> tuple[int, list[int]]:
    deadline = time.monotonic() + timeout_seconds
    observations: list[int] = []
    while True:
        current = gpu_used_mib(device)
        observations.append(current)
        if current == 0 or time.monotonic() >= deadline:
            return current, observations
        time.sleep(0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--library", type=Path, default=base.DEFAULT_LIBRARY)
    parser.add_argument("--stats", type=Path, default=base.DEFAULT_STATS)
    parser.add_argument("--srcversion", type=Path, default=base.DEFAULT_SRCVERSION)
    parser.add_argument("--fast-index-parameter", type=Path, default=base.DEFAULT_FAST_INDEX)
    parser.add_argument(
        "--direct-cutover-parameter", type=Path, default=base.DEFAULT_DIRECT_CUTOVER
    )
    parser.add_argument("--budget-parameter", type=Path, default=base.DEFAULT_BUDGET)
    parser.add_argument("--dual-parameter", type=Path, default=DEFAULT_DUAL_PARAM)
    parser.add_argument("--cache-parameter", type=Path, default=DEFAULT_CACHE_PARAM)
    parser.add_argument("--prewarm-parameter", type=Path, default=DEFAULT_PREWARM_PARAM)
    parser.add_argument("--overlap-parameter", type=Path, default=DEFAULT_OVERLAP_PARAM)
    parser.add_argument(
        "--fast-validation-parameter",
        type=Path,
        default=DEFAULT_FAST_VALIDATION_PARAM,
    )
    parser.add_argument(
        "--shadow-pte-parameter", type=Path, default=DEFAULT_SHADOW_PTE_PARAM
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=1)
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.blocks <= base.MAX_BLOCKS:
        parser.error("invalid device or block count")
    return args


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_worker(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"status": "failed"}
    try:
        require(gpu_used_mib(args.device) == 0, "GPU is not idle before probe")
        payload = run(args)
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    require(args.worker_output is not None, "missing worker output")
    write_payload(args.worker_output, payload)
    return 0 if payload.get("status") == "pass" else 1


def run_controller(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"status": "failed"}
    try:
        require(gpu_used_mib(args.device) == 0, "GPU is not idle before probe")
        with tempfile.TemporaryDirectory(prefix="uvm-dual-bank-event-") as temp:
            worker_output = Path(temp) / "worker.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-output",
                str(worker_output),
                "--library",
                str(args.library),
                "--stats",
                str(args.stats),
                "--srcversion",
                str(args.srcversion),
                "--fast-index-parameter",
                str(args.fast_index_parameter),
                "--direct-cutover-parameter",
                str(args.direct_cutover_parameter),
                "--budget-parameter",
                str(args.budget_parameter),
                "--dual-parameter",
                str(args.dual_parameter),
                "--cache-parameter",
                str(args.cache_parameter),
                "--prewarm-parameter",
                str(args.prewarm_parameter),
                "--overlap-parameter",
                str(args.overlap_parameter),
                "--fast-validation-parameter",
                str(args.fast_validation_parameter),
                "--shadow-pte-parameter",
                str(args.shadow_pte_parameter),
                "--device",
                str(args.device),
                "--blocks",
                str(args.blocks),
            ]
            completed = subprocess.run(
                command, text=True, capture_output=True, timeout=600, check=False
            )
            require(worker_output.is_file(), "worker produced no artifact")
            worker_payload = json.loads(worker_output.read_text(encoding="utf-8"))

        final_mib, observations = wait_for_zero_mib(args.device)
        payload = dict(worker_payload)
        payload["worker_returncode"] = completed.returncode
        payload["worker_stderr"] = completed.stderr
        payload["gpu_memory_final_mib"] = final_mib
        payload["gpu_memory_observations_mib"] = observations
        require(completed.returncode == 0, "dual-bank worker failed")
        require(payload.get("status") == "pass", "dual-bank worker artifact failed")
        require(final_mib == 0, "GPU memory did not return to zero")
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        try:
            final_mib, observations = wait_for_zero_mib(args.device)
            payload["gpu_memory_final_mib"] = final_mib
            payload["gpu_memory_observations_mib"] = observations
        except Exception as gpu_exc:  # noqa: BLE001
            payload["gpu_memory_final_error"] = repr(gpu_exc)

    write_payload(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "pass" else 1


def main() -> int:
    args = parse_args()
    if args.worker_output is not None:
        return run_worker(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
