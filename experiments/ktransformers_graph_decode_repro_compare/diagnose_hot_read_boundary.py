#!/usr/bin/env python3
"""Run the phase-switch benchmark with hot-weight reads around cmd99.

This is a fail-fast diagnostic wrapper, not a performance runner.  It proves
whether an ordinary hot expert remains GPU-readable immediately before and
after the static-alias destroy boundary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _hot_read(stage: str) -> None:
    import torch
    from vllm.model_executor.models import utils

    parameter_name = os.environ.get(
        "VLLM_UVM_DIAG_HOT_PARAMETER",
        "model.layers.0.mlp.experts.w13_weight",
    )
    expert_id = int(os.environ.get("VLLM_UVM_DIAG_HOT_EXPERT", "40"))
    binding = utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[parameter_name]
    tensor = binding.consumer[expert_id]
    print(
        "UVM_HOT_READ_BEGIN "
        f"stage={stage} parameter={parameter_name} expert={expert_id} "
        f"va=0x{int(tensor.data_ptr()):x} bytes={tensor.numel() * tensor.element_size()}",
        file=sys.stderr,
        flush=True,
    )
    # A reduction forces a graphics-engine read of every byte in the expert.
    checksum = tensor.view(torch.uint8).sum(dtype=torch.int64)
    value = int(checksum.item())
    torch.cuda.synchronize()
    print(
        f"UVM_HOT_READ_OK stage={stage} checksum={value}",
        file=sys.stderr,
        flush=True,
    )


def _hot_residency(stage: str) -> None:
    import torch
    from vllm.device_allocator import uvm
    from vllm.model_executor.models import utils

    parameter_name = os.environ.get(
        "VLLM_UVM_DIAG_HOT_PARAMETER",
        "model.layers.0.mlp.experts.w13_weight",
    )
    expert_id = int(os.environ.get("VLLM_UVM_DIAG_HOT_EXPERT", "40"))
    binding = utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[parameter_name]
    tensor = binding.consumer[expert_id]
    # Waiting for UVM block trackers is not a substitute for waiting for CUDA
    # work on other streams. KV initialization can still be committing native
    # pages (and evicting managed expert pages) after a tracker-only snapshot.
    # Take the diagnostic snapshot only after the whole device is quiescent so
    # that a later difference really belongs to the probe kernel itself.
    torch.cuda.synchronize()
    result = uvm.driver_batch_query_ranges_residency_bytes(
        [(int(tensor.data_ptr()), tensor.numel() * tensor.element_size())],
        device=int(torch.cuda.current_device()),
        wait_for_trackers=False,
    )
    settled = uvm.driver_batch_query_ranges_residency_bytes(
        [(int(tensor.data_ptr()), tensor.numel() * tensor.element_size())],
        device=int(torch.cuda.current_device()),
        wait_for_trackers=True,
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        "UVM_HOT_RESIDENCY "
        f"stage={stage} va=0x{int(tensor.data_ptr()):x} "
        f"gpu={result.get('gpu_bytes')} cpu={result.get('cpu_bytes')} "
        f"unknown={result.get('unknown_bytes')} free={free_bytes} "
        f"settled_gpu={settled.get('gpu_bytes')} "
        f"settled_cpu={settled.get('cpu_bytes')} "
        f"settled_unknown={settled.get('unknown_bytes')} "
        f"total={total_bytes} allocated={torch.cuda.memory_allocated()} "
        f"reserved={torch.cuda.memory_reserved()}",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    from vllm.device_allocator import uvm
    from vllm.model_executor.models import utils

    original_destroy = utils.destroy_cpu_offload_graph_static_alias
    original_remote_map = uvm.driver_prepare_ranges_remote_read_audited
    original_prefetch = uvm.batch_prefetch_ranges_to_device_with_executor
    original_rebuild = uvm.driver_rebuild_ranges_gpu_read_mappings
    remote_map_call = 0

    def diagnosed_destroy(*args, **kwargs):
        _hot_read("before_destroy")
        result = original_destroy(*args, **kwargs)
        print(
            "UVM_CMD99_RETURN "
            f"ok={result.get('ok')} state={result.get('state')} "
            f"session_id={result.get('session_id')}",
            file=sys.stderr,
            flush=True,
        )
        _hot_read("after_destroy_before_remote_map")
        return result

    utils.destroy_cpu_offload_graph_static_alias = diagnosed_destroy

    def diagnosed_remote_map(*args, **kwargs):
        nonlocal remote_map_call
        remote_map_call += 1
        call = remote_map_call
        if call >= 2:
            _hot_residency(f"before_remote_map_call_{call}")
            if os.environ.get("VLLM_UVM_DIAG_READ_BEFORE_REMOTE", "0") == "1":
                _hot_read(f"before_remote_map_call_{call}")
        if call == 2 and os.environ.get(
            "VLLM_UVM_DIAG_REMOTE_PER_RANGE", "0"
        ) == "1":
            ranges = list(args[0] if args else kwargs["ranges"])
            labels: dict[tuple[int, int], tuple[str, int]] = {}
            for name, binding in sorted(
                utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
            ):
                expert_ids = tuple(int(value) for value in (binding.expert_ids or ()))
                consumer_ranges = tuple(
                    utils._uvm_original_alias_binding_consumer_ranges(binding)
                )
                for expert_id, item in zip(expert_ids, consumer_ranges):
                    labels[(int(item[0]), int(item[1]))] = (name, expert_id)

            for index, item in enumerate(ranges):
                base, length = int(item[0]), int(item[1])
                name, expert_id = labels.get((base, length), ("unknown", -1))
                result = original_remote_map([(base, length)], **kwargs)
                print(
                    "UVM_REMOTE_RANGE_RETURN "
                    f"call={call} index={index} parameter={name} "
                    f"expert={expert_id} base=0x{base:x} length=0x{length:x} "
                    f"ok={result.get('ok')} rm_status={result.get('rm_status')}",
                    file=sys.stderr,
                    flush=True,
                )
                _hot_read(f"remote_call_{call}_range_{index}_{expert_id}")

            return {
                "ok": True,
                "reason": "diagnostic_per_range",
                "requested_count": len(ranges),
                "requested_bytes": sum(int(item[1]) for item in ranges),
                "success_count": len(ranges),
                "failure_count": 0,
                "ioctl_calls": len(ranges),
                "cpu_resident_bytes": sum(int(item[1]) for item in ranges),
                "remote_mapped_bytes": sum(int(item[1]) for item in ranges),
                "covered_bytes": sum(int(item[1]) for item in ranges),
                "unknown_bytes": 0,
                "first_failed_index": -1,
                "ioctl_ret": 0,
                "rm_status": 0,
                "errno": 0,
            }
        original_ioctl = uvm._libc_ioctl
        successful_chunk = 0

        def traced_ioctl(fd, request, params):
            nonlocal successful_chunk
            ret = original_ioctl(fd, request, params)
            if (
                call >= 2
                and os.environ.get("VLLM_UVM_DIAG_REMOTE_PER_CHUNK", "0") == "1"
                and int(request) == int(uvm._UVM_PHASE_F_BATCH_EXECUTOR_IOCTL)
                and int(getattr(params, "action", -1))
                == int(uvm.UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED)
            ):
                rm_status = int(getattr(params, "rmStatus", -1))
                count = int(getattr(params, "count", 0))
                ranges = [
                    (
                        int(params.ranges[index].base),
                        int(params.ranges[index].length),
                    )
                    for index in range(count)
                ]
                print(
                    "UVM_REMOTE_CHUNK_RETURN "
                    f"call={call} chunk={successful_chunk} ret={int(ret)} "
                    f"rm_status={rm_status} count={count} "
                    f"first=0x{ranges[0][0]:x}+0x{ranges[0][1]:x} "
                    f"last=0x{ranges[-1][0]:x}+0x{ranges[-1][1]:x}",
                    file=sys.stderr,
                    flush=True,
                )
                if int(ret) == 0 and rm_status == 0:
                    _hot_read(f"remote_call_{call}_chunk_{successful_chunk}")
                    successful_chunk += 1
            return ret

        if call >= 2 and os.environ.get(
            "VLLM_UVM_DIAG_REMOTE_PER_CHUNK", "0"
        ) == "1":
            uvm._libc_ioctl = traced_ioctl
        try:
            result = original_remote_map(*args, **kwargs)
        finally:
            uvm._libc_ioctl = original_ioctl
        print(
            "UVM_REMOTE_MAP_RETURN "
            f"call={call} ok={result.get('ok')} "
            f"requested={result.get('requested_bytes')} "
            f"mapped={result.get('remote_mapped_bytes')}",
            file=sys.stderr,
            flush=True,
        )
        if call == 1:
            _hot_read("after_remote_map_call_1")
        else:
            _hot_residency(f"after_remote_map_call_{call}_before_hot_restore")
        return result

    uvm.driver_prepare_ranges_remote_read_audited = diagnosed_remote_map

    def diagnosed_prefetch(*args, **kwargs):
        result = original_prefetch(*args, **kwargs)
        if remote_map_call >= 2:
            torch = __import__("torch")
            torch.cuda.synchronize()
            _hot_residency(f"after_hot_prefetch_call_remote_{remote_map_call}")
            if os.environ.get(
                "VLLM_UVM_DIAG_READ_AFTER_HOT_PREFETCH", "0"
            ) == "1":
                _hot_read(f"after_hot_prefetch_call_remote_{remote_map_call}")
        return result

    uvm.batch_prefetch_ranges_to_device_with_executor = diagnosed_prefetch

    def diagnosed_rebuild(*args, **kwargs):
        _hot_residency("before_parameter_mapping_rebuild")
        result = original_rebuild(*args, **kwargs)
        print(
            "UVM_PARAMETER_MAPPING_REBUILD_RETURN "
            f"ok={result.get('ok')} requested={result.get('requested_bytes')} "
            f"success={result.get('success_count')} "
            f"failure={result.get('failure_count')} "
            f"rm_status={result.get('rm_status')}",
            file=sys.stderr,
            flush=True,
        )
        _hot_residency("after_parameter_mapping_rebuild")
        if result.get("ok") is True:
            _hot_read("after_parameter_mapping_rebuild")
        return result

    uvm.driver_rebuild_ranges_gpu_read_mappings = diagnosed_rebuild

    from experiments.vllm_uvm_prefill_alias_decode_remote_graph import (
        benchmark_phase_switch_steps,
    )

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    original_profile_run = GPUModelRunner.profile_run
    original_capture_model = GPUModelRunner.capture_model
    original_hole_allocate = Worker._allocate_uvm_native_hole_sentinel

    def diagnosed_profile_run(self, *args, **kwargs):
        _hot_residency("before_profile")
        result = original_profile_run(self, *args, **kwargs)
        _hot_residency("after_profile_before_kv")
        return result

    def diagnosed_capture_model(self, *args, **kwargs):
        _hot_residency("after_kv_before_capture")
        return original_capture_model(self, *args, **kwargs)

    GPUModelRunner.profile_run = diagnosed_profile_run
    GPUModelRunner.capture_model = diagnosed_capture_model

    def diagnosed_hole_allocate(self, *args, **kwargs):
        result = original_hole_allocate(self, *args, **kwargs)
        if result is not None:
            _hot_residency("after_native_hole_before_kv")
            _hot_read("after_native_hole_before_kv")
        return result

    if os.environ.get("VLLM_UVM_DIAG_READ_AFTER_NATIVE_HOLE", "0") == "1":
        Worker._allocate_uvm_native_hole_sentinel = diagnosed_hole_allocate

    return int(benchmark_phase_switch_steps.main())


if __name__ == "__main__":
    raise SystemExit(main())
