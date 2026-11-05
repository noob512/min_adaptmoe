#!/usr/bin/env python3
"""Run one equal-capacity E6 plan sequence with full session/Graph rebuilds."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--exact-names", required=True, type=Path)
    parser.add_argument("--plan-classes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--sequence",
        default="base,d005,base,d010,base,d025,base,d050,base,d100,base",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    parser.add_argument("--cpu-offload-gb", type=float, default=30.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def load_plan_classes(path: Path) -> tuple[str, dict[str, Path], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "plan-class manifest is not an object")
    raw_classes = payload.get("classes")
    require(isinstance(raw_classes, dict) and raw_classes, "plan classes absent")
    paths: dict[str, Path] = {}
    selected_bytes: dict[str, int] = {}
    for raw_name, raw_record in raw_classes.items():
        name = str(raw_name).lower()
        require(isinstance(raw_record, dict), f"invalid class {name}")
        plan_path = (path.parent / str(raw_record.get("plan_file"))).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selected = plan.get("selected_main_bytes")
        require(isinstance(selected, int) and selected > 0, f"{name}: selected bytes")
        paths[name] = plan_path
        selected_bytes[name] = selected
    initial = str(payload.get("initial_class", "")).lower()
    require(initial in paths, "initial plan class absent")
    return initial, paths, selected_bytes


def load_prompt(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests") if isinstance(payload, dict) else None
    require(isinstance(requests, list) and requests, "fixture has no request")
    tokens = requests[0].get("prompt_token_ids")
    require(
        isinstance(tokens, list) and tokens and all(isinstance(token, int) for token in tokens),
        "fixture prompt tokens are invalid",
    )
    return [int(token) for token in tokens]


def load_names(path: Path) -> tuple[str, ...]:
    names = tuple(sorted(line.strip() for line in path.read_text().splitlines() if line.strip()))
    require(names and len(names) == len(set(names)), "exact-name list is invalid")
    return names


def model_runner_from_llm(llm: Any) -> Any:
    core = getattr(llm.llm_engine.engine_core, "engine_core", None)
    require(core is not None, "in-process EngineCore unavailable")
    return core.model_executor.driver_worker.worker.model_runner


def parameter_vas(runner: Any, names: tuple[str, ...]) -> tuple[int, ...]:
    parameters = dict(runner.get_model().named_parameters())
    require(not (set(names) - set(parameters)), "selected parameters disappeared")
    return tuple(int(parameters[name].data_ptr()) for name in names)


def output_tokens(outputs: list[Any]) -> tuple[int, ...]:
    require(len(outputs) == 1 and len(outputs[0].outputs) == 1, "output cardinality")
    return tuple(int(token) for token in outputs[0].outputs[0].token_ids)


def token_digest(tokens: tuple[int, ...]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def compact(value: Any, depth: int = 0) -> Any:
    if depth > 10:
        return "<max-depth>"
    if isinstance(value, dict):
        return {
            str(key): compact(item, depth + 1)
            for key, item in value.items()
            if str(key) not in {"entry_statuses", "page_statuses"}
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 128:
            encoded = json.dumps(value, default=str, sort_keys=True).encode()
            return {"compacted": True, "count": len(value), "sha256": hashlib.sha256(encoded).hexdigest()}
        return [compact(item, depth + 1) for item in value]
    return value


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    args = parse_args()
    sequence = tuple(item.strip().lower() for item in args.sequence.split(",") if item.strip())
    initial, plan_paths, selected_bytes = load_plan_classes(args.plan_classes.resolve())
    require(sequence and sequence[0] == initial, "sequence must start at the initial class")
    require(all(item in plan_paths for item in sequence), "sequence contains unknown class")
    require(len(set(selected_bytes.values())) == 1, "full rebuild comparison must be equal capacity")
    prompt_tokens = load_prompt(args.fixture.resolve())
    names = load_names(args.exact_names.resolve())
    configured_plan = Path(
        os.environ.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE", "")
    ).resolve()
    require(configured_plan == plan_paths[initial], "startup plan differs from initial class")
    require(
        not os.environ.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE"),
        "full rebuild baseline must not enable the fixed-class registry",
    )

    result: dict[str, Any] = {
        "schema": "eurosys_e6_full_rebuild_sequence_v1",
        "status": "loading",
        "sequence": sequence,
        "initial_class": initial,
        "selected_bytes": selected_bytes,
        "prompt_tokens": len(prompt_tokens),
        "max_tokens": args.max_tokens,
        "transition_mode": "full_session_graph_rebuild",
    }
    write_result(args.output, result)
    llm: Any = None
    runner: Any = None
    error: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        import torch
        import vllm
        from vllm import SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
        from vllm.model_executor.models import utils as offload_utils

        capture_sizes = sorted({1, len(prompt_tokens)})
        max_model_len = len(prompt_tokens) + args.max_tokens + 8
        load_started = time.perf_counter()
        llm = vllm.LLM(
            model=args.model,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=False,
            max_model_len=max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=max_model_len,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enable_dbo=False,
            compilation_config={
                "mode": "NONE",
                "cudagraph_mode": "FULL_SEPARATE",
                "cudagraph_capture_sizes": capture_sizes,
                "max_cudagraph_capture_size": max(capture_sizes),
            },
            seed=args.seed,
            disable_log_stats=True,
        )
        torch.cuda.synchronize()
        runner = model_runner_from_llm(llm)
        original_parameter_vas = parameter_vas(runner, names)
        initial_counter = int(dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"])
        require(initial_counter == 2, f"unexpected initial Graph count: {initial_counter}")
        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            min_tokens=args.max_tokens,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )
        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
        records: list[dict[str, Any]] = []
        reference_tokens: tuple[int, ...] | None = None
        expected_counter = initial_counter

        for index, class_name in enumerate(sequence):
            rebuild = None
            before_counter = int(dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"])
            if index:
                rebuild = runner.rebuild_cpu_offload_uvm_expert_plan_at_idle_boundary(
                    str(plan_paths[class_name]),
                    reason=f"e6_full_rebuild_{sequence[index - 1]}_to_{class_name}",
                    scheduler_idle=True,
                    expected_selected_bytes=selected_bytes[class_name],
                )
                require(rebuild.get("ok") is True, f"rebuild failed: {rebuild!r}")
                expected_counter += 2
            torch.cuda.synchronize()
            after_rebuild_counter = int(
                dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"]
            )
            require(after_rebuild_counter == expected_counter, "Graph recapture count mismatch")
            diagnostics = offload_utils.get_cpu_offload_graph_static_alias_diagnostics()
            require(diagnostics.get("lifecycle_state") == "sealed", "rebuild topology")
            require(
                int(diagnostics.get("selected_parameter_bytes_per_forward", -1))
                == selected_bytes[class_name],
                "selected bytes after rebuild",
            )
            require(parameter_vas(runner, names) == original_parameter_vas, "Parameter VA drift")
            request_started = time.perf_counter_ns()
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            torch.cuda.synchronize()
            request_ns = time.perf_counter_ns() - request_started
            tokens = output_tokens(outputs)
            if reference_tokens is None:
                reference_tokens = tokens
            require(tokens == reference_tokens, "full rebuild changed greedy output")
            after_request = offload_utils.get_cpu_offload_graph_static_alias_diagnostics()
            require(after_request.get("lifecycle_state") == "remote_parked", "request topology")
            require(parameter_vas(runner, names) == original_parameter_vas, "request Parameter VA drift")
            final_counter = int(dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"])
            require(final_counter == expected_counter, "request recaptured a Graph")
            driver = get_static_alias_diagnostics()
            require(driver.get("session_count") == 1, "alias session absent")
            require(not any(item.get("poisoned") for item in driver.get("sessions", ())), "poisoned session")
            records.append(
                {
                    "index": index,
                    "class_name": class_name,
                    "rebuild": compact(rebuild),
                    "graph_count_before": before_counter,
                    "graph_count_after_rebuild": after_rebuild_counter,
                    "graph_count_after_request": final_counter,
                    "request_ns": request_ns,
                    "output_token_ids": tokens,
                    "output_sha256": token_digest(tokens),
                    "selected_bytes": int(after_request["selected_parameter_bytes_per_forward"]),
                    "backing_bytes": int(after_request["backing_bytes"]),
                    "parameter_vas": original_parameter_vas,
                    "driver": compact(driver),
                }
            )
            result.update({"status": "running", "records": records})
            write_result(args.output, result)

        result.update(
            {
                "status": "ok",
                "load_seconds": time.perf_counter() - load_started,
                "capture_count_after_load": initial_counter,
                "capture_count_final": expected_counter,
                "parameter_vas": original_parameter_vas,
                "output_sha256": token_digest(reference_tokens or ()),
                "records": records,
            }
        )
    except BaseException as exc:
        error = exc
        result.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        engine_client: Any = None
        engine_shutdown = False
        if llm is not None:
            try:
                engine_client = llm.llm_engine.engine_core
                engine_client.shutdown()
                engine_shutdown = True
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        try:
            from vllm.model_executor.models.utils import release_cpu_offload_uvm_original_va_shutdown_references
            result["reference_release"] = compact(
                release_cpu_offload_uvm_original_va_shutdown_references()
            )
        except Exception as exc:
            cleanup_errors.append(repr(exc))

        # The local runner is an independent strong reference to the worker,
        # model, CUDA Graph pools, and communicator-backed objects.  Release it
        # (and the in-process core which also owns it) while process groups are
        # still valid; otherwise their C++ destructors run after distributed
        # teardown and can call a pure virtual method during interpreter exit.
        runner = None
        if engine_shutdown and engine_client is not None and hasattr(
            engine_client, "engine_core"
        ):
            engine_client.engine_core = None
        llm = None
        engine_client = None
        gc.collect()

        try:
            from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            set_graph_pool_id(None)
            cleanup_dist_env_and_memory()
            result["distributed_cleanup"] = True
        except Exception as exc:
            result["distributed_cleanup"] = False
            cleanup_errors.append(repr(exc))
        with suppress(Exception):
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        result["cleanup_errors"] = cleanup_errors
        result["controlled_hard_exit_after_cleanup"] = (
            os.environ.get(
                "VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP", "0"
            ).strip()
            == "1"
        )
        if cleanup_errors and error is None:
            result.update({"status": "failed", "error": f"cleanup failed: {cleanup_errors!r}"})
        write_result(args.output, result)
    exit_code = 0 if result.get("status") == "ok" else 1
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "record_count": len(result.get("records", ())),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if exit_code == 0 and result.get("controlled_hard_exit_after_cleanup") is True:
        # PyTorch 2.9 can abort during interpreter finalization after several
        # shared CUDA Graph pools have been retired.  Every model Graph, UVM
        # session/backing object, distributed group, and process-global tensor
        # owner has already been explicitly released and audited above.  Skip
        # only the unordered interpreter/module destructor phase; runtime or
        # cleanup failures never reach this branch with a zero exit status.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
