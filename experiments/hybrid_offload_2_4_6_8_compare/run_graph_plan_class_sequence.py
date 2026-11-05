#!/usr/bin/env python3
"""Qualify C2/C4/C6/C8 GraphExec reuse in one vLLM process."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
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
    parser.add_argument("--sequence", default="c2,c4,c6,c8,c2")
    parser.add_argument(
        "--transition-paths",
        type=Path,
        help=(
            "optional JSON map from 'old->new' to an ordered list of plan "
            "classes; every intermediate class is activated, but inference "
            "runs only after the final class"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    parser.add_argument("--cpu-offload-gb", type=float, default=30.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_plan_bytes(path: Path) -> tuple[str, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "plan-class manifest is not an object")
    classes = payload.get("classes")
    require(isinstance(classes, dict) and classes, "plan classes are absent")
    selected: dict[str, int] = {}
    for class_name, raw in classes.items():
        require(isinstance(raw, dict), f"invalid class record: {class_name}")
        plan_file = raw.get("plan_file")
        require(isinstance(plan_file, str), f"class {class_name} has no plan")
        plan_path = (path.parent / plan_file).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selected_bytes = plan.get("selected_main_bytes")
        require(
            isinstance(selected_bytes, int) and selected_bytes > 0,
            f"class {class_name} has no selected_main_bytes",
        )
        selected[str(class_name).lower()] = int(selected_bytes)
    initial = str(payload.get("initial_class", "")).lower()
    require(initial in selected, "initial plan class is not registered")
    return initial, selected


def load_prompt(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests") if isinstance(payload, dict) else None
    require(isinstance(requests, list) and requests, "fixture has no requests")
    prompt = requests[0].get("prompt_token_ids")
    require(
        isinstance(prompt, list)
        and prompt
        and all(isinstance(token, int) for token in prompt),
        "fixture prompt_token_ids are invalid",
    )
    return [int(token) for token in prompt]


def load_exact_names(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [line.strip() for line in text.splitlines() if line.strip()]
    require(isinstance(payload, list) and payload, "exact-name list is empty")
    names = tuple(sorted(str(name) for name in payload))
    require(len(names) == len(set(names)), "exact-name list has duplicates")
    return names


def load_transition_paths(
    path: Path | None,
    *,
    known_classes: set[str],
) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict) and payload, "transition paths are empty")
    paths: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_path in payload.items():
        key = str(raw_key).strip().lower()
        require(
            key.count("->") == 1,
            f"invalid transition path key: {raw_key!r}",
        )
        old_class, new_class = key.split("->")
        require(
            old_class in known_classes and new_class in known_classes,
            f"transition path contains an unknown endpoint: {raw_key!r}",
        )
        require(
            isinstance(raw_path, list) and raw_path,
            f"transition path is empty: {raw_key!r}",
        )
        normalized = tuple(str(item).strip().lower() for item in raw_path)
        require(
            all(item in known_classes for item in normalized),
            f"transition path contains an unknown class: {raw_key!r}",
        )
        require(
            normalized[-1] == new_class,
            f"transition path does not end at {new_class!r}: {raw_key!r}",
        )
        paths[key] = normalized
    return paths


def model_runner_from_llm(llm: Any) -> Any:
    client = llm.llm_engine.engine_core
    core = getattr(client, "engine_core", None)
    require(core is not None, "in-process EngineCore is unavailable")
    worker = core.model_executor.driver_worker.worker
    require(worker is not None, "driver worker is unavailable")
    return worker.model_runner


def parameter_vas(runner: Any, names: tuple[str, ...]) -> tuple[int, ...]:
    parameters = dict(runner.get_model().named_parameters())
    missing = sorted(set(names) - set(parameters))
    require(not missing, f"selected parameters disappeared: {missing[:8]!r}")
    return tuple(int(parameters[name].data_ptr()) for name in names)


def stable_va_snapshot(diagnostics: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    upload_vas = tuple(
        int(arena["upload_va"])
        for arena in diagnostics.get("arenas", ())
        if "upload_va" in arena
    )
    return {
        "parameter_vas": tuple(int(value) for value in diagnostics["original_vas"]),
        "upload_vas": upload_vas,
        "source_vas": tuple(int(value) for value in diagnostics["source_vas"]),
    }


def output_token_ids(outputs: list[Any]) -> tuple[int, ...]:
    require(len(outputs) == 1, "expected exactly one request output")
    candidates = outputs[0].outputs
    require(len(candidates) == 1, "expected exactly one completion")
    return tuple(int(token) for token in candidates[0].token_ids)


def digest_tokens(tokens: tuple[int, ...]) -> str:
    encoded = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
            return {
                "compacted": True,
                "count": len(value),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        return [compact(item, depth + 1) for item in value]
    return value


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    sequence = tuple(
        item.strip().lower() for item in args.sequence.split(",") if item.strip()
    )
    require(sequence, "plan-class sequence is empty")
    require(args.max_tokens > 0, "max-tokens must be positive")
    initial_class, selected_bytes = load_plan_bytes(args.plan_classes.resolve())
    require(all(item in selected_bytes for item in sequence), "unknown class in sequence")
    transition_paths = load_transition_paths(
        args.transition_paths.resolve() if args.transition_paths is not None else None,
        known_classes=set(selected_bytes),
    )
    prompt_tokens = load_prompt(args.fixture.resolve())
    exact_names = load_exact_names(args.exact_names.resolve())
    require(
        Path(os.environ.get(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE", ""
        )).resolve()
        == args.plan_classes.resolve(),
        "plan-class environment differs from the argument",
    )

    result: dict[str, Any] = {
        "schema": "vllm_uvm_graph_plan_class_sequence_v1",
        "status": "loading",
        "sequence": sequence,
        "initial_class": initial_class,
        "selected_bytes": selected_bytes,
        "prompt_tokens": len(prompt_tokens),
        "max_tokens": args.max_tokens,
        "transition_mode": "serial_paths" if transition_paths else "direct",
        "transition_paths": {
            key: list(value) for key, value in sorted(transition_paths.items())
        },
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("VLLM_CPU_OFFLOAD_", "VLLM_UVM_"))
            or key in {"VLLM_ATTENTION_BACKEND", "VLLM_MLA_DISABLE"}
        },
    }
    write_result(args.output, result)

    llm: Any = None
    error: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        import torch
        import vllm
        from vllm import SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.device_allocator.uvm_static_alias import (
            get_static_alias_diagnostics,
        )
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
        model = runner.model
        require(
            hasattr(model, "cudagraph_class_diagnostics"),
            "model has no CUDA Graph class registry",
        )
        diagnostics = offload_utils.get_cpu_offload_graph_static_alias_diagnostics()
        graph_diagnostics = model.cudagraph_class_diagnostics()
        expected_classes = tuple(sorted(selected_bytes))
        require(
            tuple(diagnostics["graph_certificate_classes"]) == expected_classes,
            "not every plan class has a Graph certificate",
        )
        require(
            all(
                graph_diagnostics["classes"].get(class_name, {}).get(
                    "fully_captured"
                )
                is True
                for class_name in expected_classes
            ),
            "not every plan class is fully captured",
        )
        require(
            graph_diagnostics["active_class"] == initial_class
            and diagnostics["active_plan_class"] == initial_class,
            "startup did not restore the initial class",
        )
        require(
            int(diagnostics["selected_parameter_bytes_per_forward"])
            == selected_bytes[initial_class],
            "startup logical selected bytes differ from the initial plan",
        )
        stable_envelope_bytes = int(diagnostics["stable_envelope_bytes"])
        initial_backing_bytes = int(diagnostics["backing_bytes"])
        require(
            0 < initial_backing_bytes <= stable_envelope_bytes,
            "startup physical backing is outside the stable envelope",
        )
        stable_vas = stable_va_snapshot(diagnostics)
        original_parameter_vas = parameter_vas(runner, exact_names)
        require(
            tuple(stable_vas["parameter_vas"]) == original_parameter_vas,
            "diagnostic and model Parameter VAs differ",
        )
        counter_after_load = dataclasses.asdict(compilation_counter.clone())
        fixed_capture_count = int(counter_after_load["num_cudagraph_captured"])
        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            min_tokens=args.max_tokens,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )
        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
        load_seconds = time.perf_counter() - load_started
        records: list[dict[str, Any]] = []
        class_backing_bytes = {initial_class: initial_backing_bytes}
        reference_tokens: tuple[int, ...] | None = None
        active_class = initial_class

        for index, class_name in enumerate(sequence):
            before_counter = dataclasses.asdict(compilation_counter.clone())
            switch_started = time.perf_counter_ns()
            transition_key = f"{active_class}->{class_name}"
            activation_classes = transition_paths.get(
                transition_key,
                (class_name,),
            )
            wave_activations: list[dict[str, Any]] = []
            for activation_class in activation_classes:
                wave_activation = (
                    runner.activate_uvm_expert_graph_class_at_idle_boundary(
                        activation_class,
                        scheduler_idle=True,
                        # Every measured request finishes in the prebuilt
                        # remote Decode view.  A Plan Class transaction must
                        # briefly resume the sealed Prefill view, update both
                        # frozen children, and then re-park Decode.  This is a
                        # page-table switch at the certified idle boundary;
                        # it does not recapture either CUDA Graph.
                        resume_prefill_before_switch=True,
                    )
                )
                require(
                    wave_activation.get("ok") is True,
                    f"activation failed: {wave_activation!r}",
                )
                wave_activations.append(wave_activation)
            torch.cuda.synchronize()
            switch_ns = time.perf_counter_ns() - switch_started
            activation = wave_activations[-1]
            active_class = class_name
            after_activation_counter = dataclasses.asdict(
                compilation_counter.clone()
            )
            require(
                int(after_activation_counter["num_cudagraph_captured"])
                == int(before_counter["num_cudagraph_captured"])
                == fixed_capture_count,
                "plan-class activation recaptured a CUDA Graph",
            )
            before_request = (
                offload_utils.get_cpu_offload_graph_static_alias_diagnostics()
            )
            require(before_request["active_plan_class"] == class_name, "wrong plan")
            plan_switch = activation.get("plan_switch")
            if plan_switch is None:
                activation_selected_bytes = int(
                    before_request["selected_parameter_bytes_per_forward"]
                )
                activation_backing_bytes = int(before_request["backing_bytes"])
            else:
                require(
                    plan_switch.get("ok") is True,
                    "activation returned a failed plan switch",
                )
                activation_selected_bytes = int(plan_switch["new_selected_bytes"])
                activation_backing_bytes = int(plan_switch["new_backing_bytes"])
            require(
                activation_selected_bytes
                == selected_bytes[class_name]
                == int(before_request["selected_parameter_bytes_per_forward"]),
                "active logical selected bytes differ from the plan",
            )
            active_backing_bytes = int(before_request["backing_bytes"])
            require(
                active_backing_bytes == activation_backing_bytes,
                "active physical backing differs from the activation result",
            )
            require(
                0 < active_backing_bytes <= stable_envelope_bytes,
                "active physical backing is outside the stable envelope",
            )
            known_backing_bytes = class_backing_bytes.setdefault(
                class_name, active_backing_bytes
            )
            require(
                known_backing_bytes == active_backing_bytes,
                "physical backing changed for a previously observed class",
            )
            require(
                stable_va_snapshot(before_request) == stable_vas,
                "stable envelope VA changed during activation",
            )
            request_started = time.perf_counter_ns()
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            torch.cuda.synchronize()
            request_ns = time.perf_counter_ns() - request_started
            tokens = output_token_ids(outputs)
            if reference_tokens is None:
                reference_tokens = tokens
            require(tokens == reference_tokens, "class changed greedy model output")
            after_request = (
                offload_utils.get_cpu_offload_graph_static_alias_diagnostics()
            )
            after_graphs = model.cudagraph_class_diagnostics()
            after_counter = dataclasses.asdict(compilation_counter.clone())
            require(
                int(after_counter["num_cudagraph_captured"]) == fixed_capture_count,
                "request execution lazily recaptured a CUDA Graph",
            )
            require(after_request["lifecycle_state"] == "remote_parked", "bad state")
            require(after_request["active_plan_class"] == class_name, "class drift")
            require(after_graphs["active_class"] == class_name, "Graph class drift")
            require(stable_va_snapshot(after_request) == stable_vas, "VA drift")
            require(
                int(after_request["selected_parameter_bytes_per_forward"])
                == selected_bytes[class_name]
                and int(after_request["backing_bytes"]) == active_backing_bytes,
                "logical or physical plan bytes drifted during the request",
            )
            require(
                parameter_vas(runner, exact_names) == original_parameter_vas,
                "Parameter VA drift",
            )
            driver = get_static_alias_diagnostics()
            require(driver.get("session_count") == 1, "alias session disappeared")
            require(
                not any(item.get("poisoned") for item in driver.get("sessions", ())),
                "alias session is poisoned",
            )
            records.append(
                {
                    "index": index,
                    "class_name": class_name,
                    "switch_ns": switch_ns,
                    "transition_path": list(activation_classes),
                    "transition_path_plan_wall_ns": sum(
                        int(item.get("plan_switch", {}).get("wall_ns", 0))
                        for item in wave_activations
                        if isinstance(item.get("plan_switch"), dict)
                    ),
                    "wave_activations": compact(wave_activations),
                    "request_ns": request_ns,
                    "output_token_ids": tokens,
                    "output_sha256": digest_tokens(tokens),
                    "capture_count": fixed_capture_count,
                    "selected_bytes": selected_bytes[class_name],
                    "backing_bytes": active_backing_bytes,
                    "stable_envelope_bytes": int(
                        after_request["stable_envelope_bytes"]
                    ),
                    "activation": compact(activation),
                    "graph_diagnostics": compact(after_graphs),
                    "driver": compact(driver),
                }
            )
            result.update({"status": "running", "records": records})
            write_result(args.output, result)

        result.update(
            {
                "status": "ok",
                "load_seconds": load_seconds,
                "capture_count_after_load": fixed_capture_count,
                "capture_count_final": int(
                    dataclasses.asdict(compilation_counter.clone())[
                        "num_cudagraph_captured"
                    ]
                ),
                "stable_vas": compact(stable_vas),
                "class_backing_bytes": class_backing_bytes,
                "output_sha256": digest_tokens(reference_tokens or ()),
                "records": records,
            }
        )
    except BaseException as exc:
        error = exc
        result.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if llm is not None:
            with suppress(Exception):
                llm.llm_engine.engine_core.shutdown()
        with suppress(Exception):
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        try:
            from vllm.model_executor.models.utils import (
                release_cpu_offload_uvm_original_va_shutdown_references,
            )

            release = release_cpu_offload_uvm_original_va_shutdown_references()
            result["reference_release"] = compact(release)
        except Exception as exc:
            cleanup_errors.append(repr(exc))
        llm = None
        gc.collect()
        with suppress(Exception):
            import torch

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors and error is None:
            result["status"] = "failed"
            result["error"] = f"cleanup failed: {cleanup_errors!r}"
        write_result(args.output, result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
