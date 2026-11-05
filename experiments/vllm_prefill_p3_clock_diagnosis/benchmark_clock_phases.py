#!/usr/bin/env python3
"""Run same-process vLLM batches across controlled idle and GPU-clock phases.

This is a diagnostic benchmark.  It deliberately keeps one LLM instance and
one registered P3 plan alive while varying only the idle interval and GPU clock
policy.  No subprocess, NVML snapshot, JSON serialization, or file I/O occurs
between a conditioning batch and the following measured batch.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMMON = (
    SCRIPT_DIR.parent / "broad_offload_compare" / "benchmark_custom_offload.py"
)


def load_common(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("p3_clock_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_gaps(value: str) -> list[int]:
    try:
        gaps = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gaps must be comma-separated integers") from exc
    if not gaps or any(gap < 0 or gap > 10_000 for gap in gaps):
        raise argparse.ArgumentTypeError("gaps must be in [0, 10000] ms")
    return gaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-benchmark", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--model", required=True)
    parser.add_argument("--evaluation-fixture", required=True, type=Path)
    parser.add_argument("--warmup-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--mechanism-class", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--warmup-output-tokens", type=int, required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=544)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-prompt-token-ids", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--continuous-batches", type=int, default=4)
    parser.add_argument(
        "--auto-gaps-ms",
        type=parse_gaps,
        default=parse_gaps("0,50,80,110,150,250,500,500,250,150,110,80,50,0"),
    )
    parser.add_argument(
        "--locked-gaps-ms",
        type=parse_gaps,
        default=parse_gaps("0,80,110,150,250,500,500,250,150,110,80,0"),
    )
    parser.add_argument(
        "--recovery-gaps-ms", type=parse_gaps, default=parse_gaps("110,110")
    )
    parser.add_argument(
        "--busy-gaps-ms",
        type=parse_gaps,
        default=parse_gaps("50,110,500,500,110,50"),
    )
    parser.add_argument("--locked-sm-mhz", type=int, default=2400)
    parser.add_argument("--locked-memory-mhz", type=int, default=14001)
    return parser.parse_args()


class NvmlSampler:
    NVML_CLOCK_GRAPHICS = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, index: int = 0) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlShutdown.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetPerformanceState.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.nvmlDeviceGetPerformanceState.restype = ctypes.c_int
        self.lib.nvmlDeviceGetClockInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.nvmlDeviceGetClockInfo.restype = ctypes.c_int
        self.lib.nvmlDeviceGetPowerUsage.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
        self.lib.nvmlDeviceGetTemperature.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.nvmlDeviceGetTemperature.restype = ctypes.c_int
        self.lib.nvmlDeviceGetCurrPcieLinkGeneration.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.nvmlDeviceGetCurrPcieLinkGeneration.restype = ctypes.c_int
        self.lib.nvmlDeviceGetCurrPcieLinkWidth.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.nvmlDeviceGetCurrPcieLinkWidth.restype = ctypes.c_int
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with NVML status {status}")

    def _uint(self, function: Any, *args: Any) -> int:
        value = ctypes.c_uint()
        self._check(function(self.handle, *args, ctypes.byref(value)), function.__name__)
        return int(value.value)

    def sample(self) -> dict[str, float | int]:
        return {
            "epoch": time.time(),
            "perf_counter": time.perf_counter(),
            "pstate": self._uint(self.lib.nvmlDeviceGetPerformanceState),
            "graphics_mhz": self._uint(
                self.lib.nvmlDeviceGetClockInfo, self.NVML_CLOCK_GRAPHICS
            ),
            "sm_mhz": self._uint(
                self.lib.nvmlDeviceGetClockInfo, self.NVML_CLOCK_SM
            ),
            "memory_mhz": self._uint(
                self.lib.nvmlDeviceGetClockInfo, self.NVML_CLOCK_MEM
            ),
            "power_watts": self._uint(self.lib.nvmlDeviceGetPowerUsage) / 1000.0,
            "temperature_c": self._uint(
                self.lib.nvmlDeviceGetTemperature, self.NVML_TEMPERATURE_GPU
            ),
            "pcie_generation": self._uint(
                self.lib.nvmlDeviceGetCurrPcieLinkGeneration
            ),
            "pcie_width": self._uint(self.lib.nvmlDeviceGetCurrPcieLinkWidth),
        }

    def close(self) -> None:
        self._check(self.lib.nvmlShutdown(), "nvmlShutdown")


def cpu_snapshot() -> dict[str, Any]:
    try:
        libc = ctypes.CDLL(None)
        libc.sched_getcpu.restype = ctypes.c_int
        cpu = int(libc.sched_getcpu())
    except Exception:
        cpu = None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "epoch": time.time(),
        "cpu": cpu,
        "user_seconds": float(usage.ru_utime),
        "system_seconds": float(usage.ru_stime),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
    }


def clock_command(*arguments: str) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        ["sudo", "-n", "nvidia-smi", "-i", "0", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "arguments": list(arguments),
        "started_epoch": started,
        "ended_epoch": time.time(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def lock_clocks(sm_mhz: int, memory_mhz: int) -> list[dict[str, Any]]:
    return [
        clock_command("-pm", "1"),
        clock_command("-lgc", f"{sm_mhz},{sm_mhz}"),
        clock_command("-lmc", f"{memory_mhz},{memory_mhz}"),
    ]


def reset_clocks() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for arguments in (("-rgc",), ("-rmc",), ("-pm", "0")):
        try:
            results.append(clock_command(*arguments))
        except Exception as exc:
            results.append({"arguments": list(arguments), "error": repr(exc)})
    return results


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0 or args.output_tokens <= 0:
        raise SystemExit("concurrency and output-tokens must be positive")
    if args.warmup_requests != args.concurrency:
        raise SystemExit("diagnosis requires one full-size warmup batch")
    if args.continuous_batches <= 0:
        raise SystemExit("continuous-batches must be positive")
    if args.locked_sm_mhz <= 0 or args.locked_memory_mhz <= 0:
        raise SystemExit("locked clocks must be positive")
    if not args.common_benchmark.is_file():
        raise SystemExit(f"missing common benchmark: {args.common_benchmark}")

    common = load_common(args.common_benchmark)
    eval_meta, evaluation = common.load_fixture(args.evaluation_fixture)
    warm_meta, warmup = common.load_fixture(args.warmup_fixture)
    if len(evaluation) < args.concurrency or len(warmup) < args.warmup_requests:
        raise SystemExit("fixture does not contain enough requests")
    evaluation = evaluation[: args.concurrency]
    warmup = warmup[: args.warmup_requests]
    for records, output_tokens, label in (
        (evaluation, args.output_tokens, "evaluation"),
        (warmup, args.warmup_output_tokens, "warmup"),
    ):
        if any(item.get("prompt_token_ids") is None for item in records):
            raise SystemExit(f"{label} fixture lacks frozen token IDs")
        if any(item["prompt_tokens"] + output_tokens > args.max_model_len for item in records):
            raise SystemExit(f"{label} exceeds max-model-len")

    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    def engine_inputs(records: list[dict[str, Any]]) -> list[TokensPrompt]:
        return [
            TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
            for item in records
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_epoch = time.time()
    load_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        cpu_offload_gb=args.cpu_offload_gb,
        enable_prefix_caching=False,
        seed=args.seed,
        disable_log_stats=False,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    sampler = NvmlSampler(0)
    sampling = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=args.output_tokens,
        max_tokens=args.output_tokens,
    )
    warm_sampling = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=args.warmup_output_tokens,
        max_tokens=args.warmup_output_tokens,
    )
    calls: list[dict[str, Any]] = []
    clock_events: list[dict[str, Any]] = []
    call_index = 0
    clocks_locked = False

    def run_batch(
        *,
        phase: str,
        role: str,
        clock_mode: str,
        requested_gap_ms: int,
        idle_anchor: float | None,
        idle_mode: str = "sleep",
    ) -> float:
        nonlocal call_index
        if requested_gap_ms:
            if idle_mode == "sleep":
                time.sleep(requested_gap_ms / 1000.0)
            elif idle_mode == "busy":
                deadline = time.perf_counter() + requested_gap_ms / 1000.0
                while time.perf_counter() < deadline:
                    pass
            else:
                raise RuntimeError(f"unknown idle mode: {idle_mode}")
        call_start_perf = time.perf_counter()
        actual_idle_ms = (
            (call_start_perf - idle_anchor) * 1000.0 if idle_anchor is not None else None
        )
        clock_before = sampler.sample()
        cpu_before = cpu_snapshot()
        start_epoch = time.time()
        started = time.perf_counter()
        outputs = llm.generate(engine_inputs(evaluation), sampling, use_tqdm=False)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000.0
        end_epoch = time.time()
        clock_after = sampler.sample()
        cpu_after = cpu_snapshot()
        if len(outputs) != args.concurrency:
            raise RuntimeError(f"{phase}/{role}: incomplete output batch")
        records = [
            common.request_record(output, expected)
            for output, expected in zip(outputs, evaluation)
        ]
        if any(record["prompt_token_ids_exact"] is not True for record in records):
            raise RuntimeError(f"{phase}/{role}: prompt token IDs changed")
        if any(record["output_tokens"] != args.output_tokens for record in records):
            raise RuntimeError(f"{phase}/{role}: output token count changed")
        calls.append(
            {
                "call_index": call_index,
                "phase": phase,
                "role": role,
                "clock_mode": clock_mode,
                "requested_gap_ms": requested_gap_ms,
                "actual_idle_ms": actual_idle_ms,
                "idle_mode": idle_mode,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "wall_ms": wall_ms,
                "clock_before": clock_before,
                "clock_after": clock_after,
                "cpu_before": cpu_before,
                "cpu_after": cpu_after,
                "ttft_ms": common.summarize(
                    [float(record["ttft_ms"]) for record in records]
                ),
                "output_token_ids_sha256": common.sha256_json(
                    [record["output_token_ids"] for record in records]
                ),
            }
        )
        call_index += 1
        return time.perf_counter()

    warmup_started_epoch = time.time()
    warm_started = time.perf_counter()
    warm_outputs = llm.generate(
        engine_inputs(warmup), warm_sampling, use_tqdm=False
    )
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warm_started
    warmup_end_epoch = time.time()
    if len(warm_outputs) != args.warmup_requests:
        raise RuntimeError("warmup output batch is incomplete")
    anchor = time.perf_counter()
    try:
        for _ in range(args.continuous_batches):
            anchor = run_batch(
                phase="auto_contiguous",
                role="measure",
                clock_mode="auto",
                requested_gap_ms=0,
                idle_anchor=anchor,
            )

        for gap_ms in args.auto_gaps_ms:
            anchor = run_batch(
                phase="auto_gap_sweep",
                role="condition",
                clock_mode="auto",
                requested_gap_ms=0,
                idle_anchor=anchor,
            )
            anchor = run_batch(
                phase="auto_gap_sweep",
                role="measure",
                clock_mode="auto",
                requested_gap_ms=gap_ms,
                idle_anchor=anchor,
            )

        for gap_ms in args.busy_gaps_ms:
            anchor = run_batch(
                phase="auto_busy_gap_sweep",
                role="condition",
                clock_mode="auto",
                requested_gap_ms=0,
                idle_anchor=anchor,
            )
            anchor = run_batch(
                phase="auto_busy_gap_sweep",
                role="measure",
                clock_mode="auto",
                requested_gap_ms=gap_ms,
                idle_anchor=anchor,
                idle_mode="busy",
            )

        clocks_locked = True
        clock_events.extend(lock_clocks(args.locked_sm_mhz, args.locked_memory_mhz))
        anchor = time.perf_counter()
        for gap_ms in args.locked_gaps_ms:
            anchor = run_batch(
                phase="locked_gap_sweep",
                role="condition",
                clock_mode="locked",
                requested_gap_ms=0,
                idle_anchor=anchor,
            )
            anchor = run_batch(
                phase="locked_gap_sweep",
                role="measure",
                clock_mode="locked",
                requested_gap_ms=gap_ms,
                idle_anchor=anchor,
            )

        clock_events.extend(reset_clocks())
        clocks_locked = False
        anchor = time.perf_counter()
        for gap_ms in args.recovery_gaps_ms:
            anchor = run_batch(
                phase="auto_recovery",
                role="condition",
                clock_mode="auto_recovery",
                requested_gap_ms=0,
                idle_anchor=anchor,
            )
            anchor = run_batch(
                phase="auto_recovery",
                role="measure",
                clock_mode="auto_recovery",
                requested_gap_ms=gap_ms,
                idle_anchor=anchor,
            )
    finally:
        if clocks_locked:
            clock_events.extend(reset_clocks())

    sampler.close()
    shutdown_started_epoch = time.time()
    llm.llm_engine.engine_core.shutdown()
    shutdown_end_epoch = time.time()
    output_hashes = {call["output_token_ids_sha256"] for call in calls}
    if len(output_hashes) != 1:
        raise RuntimeError("repeated evaluation batches produced different tokens")

    result = {
        "schema": "vllm_prefill_p3_clock_diagnosis_v1",
        "status": "ok",
        "profile_label": args.profile_label,
        "mechanism_class": args.mechanism_class,
        "started_epoch": started_epoch,
        "config": {
            **vars(args),
            "common_benchmark": str(args.common_benchmark.resolve()),
            "evaluation_fixture": str(args.evaluation_fixture.resolve()),
            "warmup_fixture": str(args.warmup_fixture.resolve()),
            "output": str(args.output.resolve()),
            "auto_gaps_ms": args.auto_gaps_ms,
            "locked_gaps_ms": args.locked_gaps_ms,
            "recovery_gaps_ms": args.recovery_gaps_ms,
            "busy_gaps_ms": args.busy_gaps_ms,
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "enable_prefix_caching": False,
        },
        "versions": {
            "vllm": getattr(vllm, "__version__", None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "fixtures": {
            "evaluation_sha256": common.sha256_file(args.evaluation_fixture),
            "evaluation_metadata": eval_meta,
            "warmup_sha256": common.sha256_file(args.warmup_fixture),
            "warmup_metadata": warm_meta,
        },
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("VLLM_UVM_", "VLLM_CPU_OFFLOAD_"))
            or key in {"VLLM_USE_UVM", "CUDA_VISIBLE_DEVICES"}
        },
        "load_seconds": load_seconds,
        "warmup": {
            "start_epoch": warmup_started_epoch,
            "end_epoch": warmup_end_epoch,
            "elapsed_seconds": warmup_seconds,
            "requests": args.warmup_requests,
        },
        "calls": calls,
        "clock_events": clock_events,
        "output_token_ids_sha256": next(iter(output_hashes)),
        "shutdown_started_epoch": shutdown_started_epoch,
        "shutdown_end_epoch": shutdown_end_epoch,
        "process_memory": common.process_memory_snapshot(),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
