#!/usr/bin/env python3
"""Qualify one MoE-Infinity batch=1 profile in one cold process.

This is deliberately a single-case runner.  The shell orchestrator launches
each profile in a fresh process so model/cache state cannot leak between
profiles.  It uses the HuggingFace generation path because the current native
batch=1 path does not forward ``streamer`` callbacks needed for TTFT/TPOT.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable


MODES: dict[str, dict[str, Any]] = {
    "demand-only": {
        "speculative_prefetch": False,
        "speculative_prefetch_overlap": False,
        "correction_layer_mode": "upstream",
    },
    "upstream-speculative": {
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": False,
        "correction_layer_mode": "upstream",
    },
    "fixed-speculative": {
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": False,
        "correction_layer_mode": "current",
    },
    "overlap": {
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": True,
        "correction_layer_mode": "current",
    },
}

DEFAULT_PROMPT = (
    "A careful systems evaluation separates model loading, prompt prefill, "
    "and autoregressive decoding. It records exact token identifiers, memory "
    "budgets, cache behavior, and all transfer decisions before comparing "
    "latency or throughput. The following passage is repeated only to create "
    "a deterministic prompt of an exact token length for a reproducible "
    "single-request benchmark. "
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_sha256(token_ids: list[int]) -> str:
    # The representation is intentionally simple and language-independent.
    payload = ",".join(str(value) for value in token_ids).encode("ascii")
    return sha256_bytes(payload)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def command_bytes(command: list[str], cwd: Path | None = None) -> bytes | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def numeric_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in after.items():
        old_value = before.get(key)
        if isinstance(value, dict) and isinstance(old_value, dict):
            delta[key] = numeric_delta(value, old_value)
        elif isinstance(value, int) and isinstance(old_value, int):
            delta[key] = value - old_value
    return delta


class ArcherEngineProxy:
    """Count real Python-to-pybind enqueue_prefetch calls."""

    def __init__(self, target: Any, owner: "PrefetchInstrumentation") -> None:
        self._target = target
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def enqueue_prefetch(self, tensor_id: int, gpu_id: int) -> Any:
        owner = self._owner
        owner.enqueue_attempts += 1
        source = owner.current_source or "unknown"
        owner.enqueue_by_source[source] = owner.enqueue_by_source.get(source, 0) + 1
        owner.unique_tensor_ids.add(int(tensor_id))
        owner.add_sample(
            "enqueue",
            {
                "source": source,
                "tensor_id": int(tensor_id),
                "gpu_id": int(gpu_id),
            },
        )
        try:
            result = self._target.enqueue_prefetch(tensor_id, gpu_id)
        except BaseException:
            owner.enqueue_errors += 1
            raise
        owner.enqueue_successes += 1
        return result


class PrefetchInstrumentation:
    """Instance-local wrappers; vendor source files are not modified."""

    def __init__(
        self,
        prefetcher: Any,
        correction_layer_mode: str,
        sample_limit: int,
    ) -> None:
        if correction_layer_mode not in {"upstream", "current"}:
            raise ValueError(f"unknown correction layer mode: {correction_layer_mode}")
        self.prefetcher = prefetcher
        self.correction_layer_mode = correction_layer_mode
        self.sample_limit = max(0, sample_limit)
        self.samples: list[dict[str, Any]] = []
        self.current_source: str | None = None
        self.prediction_calls = 0
        self.prediction_nonempty = 0
        self.predicted_experts_total = 0
        self.correction_calls = 0
        self.correction_actual_experts_total = 0
        self.correction_misses_total = 0
        self.requested_effective_layer_differences = 0
        self.enqueue_attempts = 0
        self.enqueue_successes = 0
        self.enqueue_errors = 0
        self.enqueue_by_source: dict[str, int] = {}
        self.unique_tensor_ids: set[int] = set()

        self._original_engine = prefetcher.archer_engine
        self._original_speculative: Callable[..., Any] = prefetcher.speculative_prefetch
        self._original_correct: Callable[..., Any] = prefetcher.correct_prefetch
        prefetcher.archer_engine = ArcherEngineProxy(self._original_engine, self)

        def speculative_wrapper(
            instance: Any, layer_idx: int, router_logits: Any | None = None
        ) -> Any:
            del instance
            self.prediction_calls += 1
            previous_source = self.current_source
            self.current_source = "prediction"
            try:
                result = self._original_speculative(layer_idx, router_logits)
            finally:
                self.current_source = previous_source
            predicted = sorted(
                int(value)
                for value in getattr(
                    self.prefetcher, "_last_speculative_prediction", set()
                )
            )
            if predicted:
                self.prediction_nonempty += 1
            self.predicted_experts_total += len(predicted)
            self.add_sample(
                "prediction",
                {
                    "source_layer": int(layer_idx),
                    "target_layer": int(layer_idx) + 1,
                    "predicted_experts": predicted,
                },
            )
            return result

        def correction_wrapper(
            instance: Any,
            layer_idx: int,
            actual_expert_ids: list[int],
            predicted_expert_ids: set[int] | None = None,
        ) -> Any:
            del instance
            requested_layer = int(layer_idx)
            effective_layer = requested_layer
            if self.correction_layer_mode == "current":
                effective_layer = max(0, requested_layer - 1)
            if requested_layer != effective_layer:
                self.requested_effective_layer_differences += 1

            predicted = (
                set(int(value) for value in predicted_expert_ids)
                if predicted_expert_ids is not None
                else set(
                    int(value)
                    for value in getattr(
                        self.prefetcher, "_last_speculative_prediction", set()
                    )
                )
            )
            actual = [int(value) for value in actual_expert_ids]
            missed = [value for value in actual if value not in predicted]
            self.correction_calls += 1
            self.correction_actual_experts_total += len(actual)
            self.correction_misses_total += len(missed)
            self.add_sample(
                "correction",
                {
                    "requested_layer": requested_layer,
                    "effective_layer": effective_layer,
                    "actual_experts": actual,
                    "predicted_experts": sorted(predicted),
                    "missed_experts": missed,
                },
            )

            previous_source = self.current_source
            self.current_source = "correction"
            try:
                return self._original_correct(
                    effective_layer, actual, predicted_expert_ids
                )
            finally:
                self.current_source = previous_source

        prefetcher.speculative_prefetch = types.MethodType(
            speculative_wrapper, prefetcher
        )
        prefetcher.correct_prefetch = types.MethodType(correction_wrapper, prefetcher)

    def add_sample(self, kind: str, values: dict[str, Any]) -> None:
        if len(self.samples) < self.sample_limit:
            self.samples.append({"kind": kind, **values})

    def snapshot(self) -> dict[str, Any]:
        return {
            "prediction_calls": self.prediction_calls,
            "prediction_nonempty": self.prediction_nonempty,
            "predicted_experts_total": self.predicted_experts_total,
            "correction_calls": self.correction_calls,
            "correction_actual_experts_total": self.correction_actual_experts_total,
            "correction_misses_total": self.correction_misses_total,
            "requested_effective_layer_differences": (
                self.requested_effective_layer_differences
            ),
            "enqueue_attempts": self.enqueue_attempts,
            "enqueue_successes": self.enqueue_successes,
            "enqueue_errors": self.enqueue_errors,
            "enqueue_by_source": copy.deepcopy(self.enqueue_by_source),
            "unique_tensor_ids": len(self.unique_tensor_ids),
            "sample_count": len(self.samples),
        }


class TokenTimingStreamer:
    """Minimal HF streamer recording one batch=1 token callback at a time."""

    def __init__(self, torch_module: Any, expected_prompt_tokens: int) -> None:
        self.torch = torch_module
        self.expected_prompt_tokens = expected_prompt_tokens
        self.put_calls = 0
        self.prompt_shape: list[int] | None = None
        self.prompt_token_count: int | None = None
        self.generated_token_ids: list[int] = []
        self.generated_token_times: list[float] = []
        self.end_called = False
        self.end_time: float | None = None

    def put(self, value: Any) -> None:
        self.torch.cuda.synchronize()
        now = time.perf_counter()
        detached = value.detach().to(device="cpu")
        flat = [int(item) for item in detached.reshape(-1).tolist()]
        if self.put_calls == 0:
            self.prompt_shape = list(detached.shape)
            self.prompt_token_count = len(flat)
            if self.prompt_token_count != self.expected_prompt_tokens:
                raise RuntimeError(
                    "streamer prompt callback contained "
                    f"{self.prompt_token_count} tokens, expected "
                    f"{self.expected_prompt_tokens}"
                )
        else:
            if len(flat) != 1:
                raise RuntimeError(
                    f"batch=1 streamer callback contained {len(flat)} generated tokens"
                )
            self.generated_token_ids.append(flat[0])
            self.generated_token_times.append(now)
        self.put_calls += 1

    def end(self) -> None:
        self.torch.cuda.synchronize()
        self.end_called = True
        self.end_time = time.perf_counter()


def build_exact_prompt(
    tokenizer: Any,
    prompt_length: int,
    prompt_text: str | None,
    prompt_file: Path | None,
    prompt_token_ids_file: Path | None,
) -> tuple[list[int], dict[str, Any]]:
    selected_sources = sum(
        value is not None
        for value in (prompt_text, prompt_file, prompt_token_ids_file)
    )
    if selected_sources > 1:
        raise ValueError(
            "--prompt-text, --prompt-file and --prompt-token-ids-file are "
            "mutually exclusive"
        )
    if prompt_token_ids_file is not None:
        payload = json.loads(prompt_token_ids_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("prompt_token_ids")
        if not isinstance(payload, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in payload
        ):
            raise ValueError(
                "prompt token ID file must be a JSON integer list or an object "
                "with prompt_token_ids"
            )
        prompt_ids = [int(value) for value in payload]
        if len(prompt_ids) != prompt_length:
            raise ValueError(
                f"prompt token ID file has {len(prompt_ids)} IDs, expected "
                f"{prompt_length}"
            )
        return prompt_ids, {
            "source": str(prompt_token_ids_file.resolve()),
            "source_file_sha256": sha256_file(prompt_token_ids_file),
            "seed_text_sha256": None,
            "seed_token_count": None,
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": token_sha256(prompt_ids),
            "token_hash_encoding": "ascii comma-separated decimal token IDs",
        }
    if prompt_file is not None:
        seed_text = prompt_file.read_text(encoding="utf-8")
        source = str(prompt_file.resolve())
    elif prompt_text is not None:
        seed_text = prompt_text
        source = "cli"
    else:
        seed_text = DEFAULT_PROMPT
        source = "built-in-v1"
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("prompt seed tokenized to an empty sequence")
    repeats = (prompt_length + len(seed_ids) - 1) // len(seed_ids)
    prompt_ids = (seed_ids * repeats)[:prompt_length]
    if len(prompt_ids) != prompt_length:
        raise AssertionError("internal exact-prompt construction failure")
    return prompt_ids, {
        "source": source,
        "seed_text_sha256": sha256_bytes(seed_text.encode("utf-8")),
        "seed_token_count": len(seed_ids),
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": token_sha256(prompt_ids),
        "token_hash_encoding": "ascii comma-separated decimal token IDs",
    }


def run_generation(
    *,
    torch_module: Any,
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    attention_mask: Any,
    prompt_length: int,
    output_tokens: int,
    instrumentation: PrefetchInstrumentation,
    trial_index: int,
    warmup: bool,
) -> dict[str, Any]:
    before = instrumentation.snapshot()
    streamer = TokenTimingStreamer(torch_module, prompt_length)
    torch_module.cuda.reset_peak_memory_stats()
    torch_module.cuda.synchronize()
    started = time.perf_counter()
    with torch_module.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=output_tokens,
            min_new_tokens=output_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch_module.cuda.synchronize()
    finished = time.perf_counter()
    after = instrumentation.snapshot()

    if list(input_ids.shape) != [1, prompt_length]:
        raise RuntimeError(f"input shape drifted to {list(input_ids.shape)}")
    generated = [int(item) for item in output_ids[0, prompt_length:].tolist()]
    if len(generated) != output_tokens:
        raise RuntimeError(
            f"generated {len(generated)} tokens, expected {output_tokens}"
        )
    if streamer.generated_token_ids != generated:
        raise RuntimeError("streamer token IDs differ from returned generation")
    if len(streamer.generated_token_times) != output_tokens:
        raise RuntimeError(
            "streamer recorded "
            f"{len(streamer.generated_token_times)} tokens, expected {output_tokens}"
        )
    if not streamer.end_called:
        raise RuntimeError("streamer.end() was not called")

    first_token_time = streamer.generated_token_times[0]
    ttft = first_token_time - started
    decode_denominator = max(1, output_tokens - 1)
    decode_tpot = (finished - first_token_time) / decode_denominator
    observed_itl = (
        (streamer.generated_token_times[-1] - first_token_time)
        / decode_denominator
        if output_tokens > 1
        else 0.0
    )
    intervals = [
        later - earlier
        for earlier, later in zip(
            streamer.generated_token_times,
            streamer.generated_token_times[1:],
        )
    ]
    return {
        "trial_index": trial_index,
        "warmup": warmup,
        "prompt_tokens": prompt_length,
        "output_tokens": len(generated),
        "e2e_seconds": finished - started,
        "ttft_seconds": ttft,
        "decode_tpot_seconds": decode_tpot,
        "observed_inter_token_seconds": observed_itl,
        "decode_tokens_per_second": 1.0 / decode_tpot if decode_tpot > 0 else None,
        "output_tokens_per_e2e_second": output_tokens / (finished - started),
        "inter_token_intervals_seconds": intervals,
        "generated_token_ids": generated,
        "generated_token_ids_sha256": token_sha256(generated),
        "decoded_text": tokenizer.decode(generated, skip_special_tokens=False),
        "streamer": {
            "put_calls": streamer.put_calls,
            "prompt_shape": streamer.prompt_shape,
            "prompt_token_count": streamer.prompt_token_count,
            "generated_callbacks": len(streamer.generated_token_times),
            "end_called": streamer.end_called,
            "end_to_main_finish_seconds": (
                finished - streamer.end_time if streamer.end_time is not None else None
            ),
        },
        "cuda_memory": {
            "allocated_bytes_after": int(torch_module.cuda.memory_allocated()),
            "reserved_bytes_after": int(torch_module.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
        },
        "prefetch_counters_before": before,
        "prefetch_counters_after": after,
        "prefetch_counter_delta": numeric_delta(after, before),
    }


def qualification_checks(
    mode: str,
    mode_config: dict[str, Any],
    warmups: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    instrumentation: PrefetchInstrumentation,
    executor_overlap: bool,
    expected_prompt_tokens: int,
    expected_output_tokens: int,
    expected_warmups: int,
    expected_repetitions: int,
    require_cross_phase_output_match: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    measured_deltas = [trial["prefetch_counter_delta"] for trial in trials]

    def total(field: str) -> int:
        return sum(int(delta.get(field, 0)) for delta in measured_deltas)

    if len(warmups) != expected_warmups:
        failures.append(
            f"recorded {len(warmups)} warmups, expected {expected_warmups}"
        )
    if len(trials) != expected_repetitions:
        failures.append(
            f"recorded {len(trials)} repetitions, expected {expected_repetitions}"
        )

    all_records = [*warmups, *trials]
    for expected_warmup, records in ((True, warmups), (False, trials)):
        for expected_index, record in enumerate(records):
            if record.get("warmup") is not expected_warmup:
                failures.append(
                    f"record {expected_index} warmup flag is {record.get('warmup')!r}, "
                    f"expected {expected_warmup}"
                )
            if record.get("trial_index") != expected_index:
                failures.append(
                    f"record trial_index={record.get('trial_index')!r}, expected "
                    f"{expected_index}"
                )
            if record.get("prompt_tokens") != expected_prompt_tokens:
                failures.append(
                    f"record {expected_index} prompt_tokens="
                    f"{record.get('prompt_tokens')!r}, expected {expected_prompt_tokens}"
                )
            if record.get("output_tokens") != expected_output_tokens:
                failures.append(
                    f"record {expected_index} output_tokens="
                    f"{record.get('output_tokens')!r}, expected {expected_output_tokens}"
                )

    hashes = [trial["generated_token_ids_sha256"] for trial in trials]
    stable_hash = len(set(hashes)) == 1
    if not stable_hash:
        failures.append("measured repetitions produced different output token hashes")
    all_hashes = [record["generated_token_ids_sha256"] for record in all_records]
    all_outputs_stable = bool(all_hashes) and len(set(all_hashes)) == 1
    warmup_hashes = [record["generated_token_ids_sha256"] for record in warmups]
    warmup_outputs_stable = bool(warmup_hashes) and len(set(warmup_hashes)) == 1
    if not warmup_outputs_stable:
        failures.append("warmup requests produced different output token hashes")
    if require_cross_phase_output_match and not all_outputs_stable:
        failures.append("warmup and measured requests produced different output hashes")
    enqueue_errors = total("enqueue_errors")
    if enqueue_errors:
        failures.append(f"observed {enqueue_errors} enqueue_prefetch errors")

    expected_speculative = bool(mode_config["speculative_prefetch"])
    prediction_calls = total("prediction_calls")
    correction_calls = total("correction_calls")
    enqueue_attempts = total("enqueue_attempts")
    layer_differences = total("requested_effective_layer_differences")
    final_counters = instrumentation.snapshot()
    lifecycle_enqueue_errors = int(final_counters.get("enqueue_errors", 0))
    if lifecycle_enqueue_errors:
        failures.append(
            "warmup+measurement lifecycle observed "
            f"{lifecycle_enqueue_errors} enqueue_prefetch errors"
        )
    if expected_speculative:
        if prediction_calls <= 0:
            failures.append("speculative profile made no prediction calls")
        if correction_calls <= 0:
            failures.append("speculative profile made no correction calls")
        if enqueue_attempts <= 0:
            failures.append("speculative profile made no enqueue_prefetch calls")
    else:
        lifecycle_nonzero = {
            "prediction_calls": int(final_counters.get("prediction_calls", 0)),
            "correction_calls": int(final_counters.get("correction_calls", 0)),
            "enqueue_attempts": int(final_counters.get("enqueue_attempts", 0)),
        }
        if any(lifecycle_nonzero.values()):
            failures.append(
                "demand-only unexpectedly used prefetch path during warmup or "
                f"measurement: {lifecycle_nonzero}"
            )

    expected_current_layer = mode_config["correction_layer_mode"] == "current"
    if expected_speculative and expected_current_layer and layer_differences <= 0:
        failures.append("current-layer monkeypatch did not alter any requested layer")
    if not expected_current_layer and layer_differences != 0:
        failures.append("upstream profile unexpectedly altered correction layers")

    expected_overlap = bool(mode_config["speculative_prefetch_overlap"])
    if executor_overlap != expected_overlap:
        failures.append(
            f"executor overlap={executor_overlap}, expected {expected_overlap}"
        )
    checks = {
        "mode": mode,
        "expected_speculative_prefetch": expected_speculative,
        "expected_overlap": expected_overlap,
        "executor_overlap": executor_overlap,
        "expected_correction_layer_mode": mode_config["correction_layer_mode"],
        "output_hash_stable": stable_hash,
        "warmup_output_hash_stable": warmup_outputs_stable,
        "cross_phase_output_match_required": require_cross_phase_output_match,
        "warmup_and_measured_output_hash_stable": all_outputs_stable,
        "warmup_and_measured_output_hashes": all_hashes,
        "recorded_warmups": len(warmups),
        "recorded_repetitions": len(trials),
        "measured_output_hashes": hashes,
        "measured_prediction_calls": prediction_calls,
        "measured_correction_calls": correction_calls,
        "measured_enqueue_attempts": enqueue_attempts,
        "measured_enqueue_errors": enqueue_errors,
        "measured_requested_effective_layer_differences": layer_differences,
        "instrumentation_final": final_counters,
        "status": "ok" if not failures else "error",
    }
    return checks, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--offload-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-token-ids-file", type=Path)
    parser.add_argument("--warmup-prompt-token-ids-file", type=Path)
    parser.add_argument("--event-sample-limit", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prompt_length <= 0 or args.output_tokens <= 0:
        raise SystemExit("prompt/output token counts must be positive")
    if args.warmup_requests < 1:
        raise SystemExit("at least one warmup request is required for qualification")
    if args.repetitions < 2:
        raise SystemExit("at least two measured repetitions are required")
    if not 0.0 < args.device_memory_ratio < 1.0:
        raise SystemExit("--device-memory-ratio must be in (0, 1)")
    if args.num_threads <= 0:
        raise SystemExit("--num-threads must be positive")

    args.offload_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode_config = MODES[args.mode]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "moe_infinity_public_artifact_qualification",
        "status": "initializing",
        "mode": args.mode,
        "mode_config": mode_config,
        "model": args.model,
        "offload_dir": str(args.offload_dir.resolve()),
        "protocol": {
            "batch_size": 1,
            "prompt_tokens": args.prompt_length,
            "output_tokens": args.output_tokens,
            "warmup_requests": args.warmup_requests,
            "measured_repetitions": args.repetitions,
            "expert_compute_threads": args.num_threads,
            "device_memory_ratio": args.device_memory_ratio,
            "greedy": True,
            "use_native_engine": False,
            "primary_decode_tpot_definition": (
                "(main generation finish - first generated token callback) / "
                "(output_tokens - 1)"
            ),
        },
    }
    atomic_json(args.output, result)

    try:
        import torch
        import transformers
        from moe_infinity import MoE
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        vendor_root = Path(__file__).resolve().parents[2] / "MoE-Infinity"
        extension = vendor_root / "moe_infinity" / (
            f"_store.cpython-{os.sys.version_info.major}{os.sys.version_info.minor}"
            "-x86_64-linux-gnu.so"
        )
        dirty_status = command_output(
            ["git", "status", "--porcelain=v1"], cwd=vendor_root
        )
        diff = command_bytes(["git", "diff", "--binary"], cwd=vendor_root)
        result["environment"] = {
            "python": os.sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_count": torch.cuda.device_count(),
            "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "moe_infinity_commit": command_output(
                ["git", "rev-parse", "HEAD"], cwd=vendor_root
            ),
            "moe_infinity_status_porcelain": dirty_status,
            "moe_infinity_diff_sha256": (
                sha256_bytes(diff) if diff is not None else None
            ),
            "store_extension": str(extension),
            "store_extension_sha256": sha256_file(extension),
            "driver": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,pci.link.gen.current,pci.link.width.current",
                    "--format=csv,noheader,nounits",
                ]
            ),
        }

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt_ids, prompt_metadata = build_exact_prompt(
            tokenizer,
            args.prompt_length,
            args.prompt_text,
            args.prompt_file,
            args.prompt_token_ids_file,
        )
        tokenizer_size = len(tokenizer)
        if any(value >= tokenizer_size for value in prompt_ids):
            raise ValueError(
                f"prompt contains a token ID outside tokenizer size {tokenizer_size}"
            )
        prompt_metadata["tokenizer_size"] = tokenizer_size
        result["prompt"] = prompt_metadata
        warmup_prompt_ids = prompt_ids
        warmup_prompt_metadata = prompt_metadata
        if args.warmup_prompt_token_ids_file is not None:
            warmup_prompt_ids, warmup_prompt_metadata = build_exact_prompt(
                tokenizer,
                args.prompt_length,
                None,
                None,
                args.warmup_prompt_token_ids_file,
            )
        result["warmup_prompt"] = warmup_prompt_metadata

        config = {
            "offload_path": str(args.offload_dir.resolve()),
            "device_memory_ratio": args.device_memory_ratio,
            "num_threads": args.num_threads,
            "use_native_engine": False,
            "prefetch": False,
            "speculative_prefetch": mode_config["speculative_prefetch"],
            "speculative_prefetch_overlap": mode_config[
                "speculative_prefetch_overlap"
            ],
        }
        load_started = time.perf_counter()
        model = MoE(args.model, config)
        torch.cuda.synchronize()
        result["load_seconds"] = time.perf_counter() - load_started
        result["resolved_archer_config"] = {
            "offload_path": str(Path(model.engine_config.offload_path).resolve()),
            "device_memory_ratio": model.engine_config.device_memory_ratio,
            "num_threads": model.engine_config.num_threads,
            "use_native_engine": model.engine_config.use_native_engine,
            "prefetch": model.engine_config.prefetch,
            "speculative_prefetch": model.engine_config.speculative_prefetch,
            "speculative_prefetch_overlap": (
                model.engine_config.speculative_prefetch_overlap
            ),
        }
        expected_resolved_config = {
            "offload_path": str(args.offload_dir.resolve()),
            "device_memory_ratio": args.device_memory_ratio,
            "num_threads": args.num_threads,
            "use_native_engine": False,
            "prefetch": False,
            "speculative_prefetch": mode_config["speculative_prefetch"],
            "speculative_prefetch_overlap": mode_config[
                "speculative_prefetch_overlap"
            ],
        }
        if result["resolved_archer_config"] != expected_resolved_config:
            raise RuntimeError(
                "resolved ArcherConfig differs from requested config: "
                f"resolved={result['resolved_archer_config']!r}, "
                f"expected={expected_resolved_config!r}"
            )
        if model.use_native_engine:
            raise RuntimeError("model unexpectedly selected native generation engine")

        prefetcher = model.engine.expert_prefetcher
        executor = model.engine.expert_executor
        if prefetcher is None or executor is None:
            raise RuntimeError("MoE-Infinity did not construct prefetcher/executor")
        instrumentation = PrefetchInstrumentation(
            prefetcher,
            correction_layer_mode=mode_config["correction_layer_mode"],
            sample_limit=args.event_sample_limit,
        )

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones_like(input_ids)
        warmup_input_ids = torch.tensor(
            [warmup_prompt_ids], dtype=torch.long, device="cuda:0"
        )
        warmup_attention_mask = torch.ones_like(warmup_input_ids)
        if tuple(input_ids.shape) != (1, args.prompt_length):
            raise AssertionError(f"unexpected input shape {tuple(input_ids.shape)}")
        if tuple(warmup_input_ids.shape) != (1, args.prompt_length):
            raise AssertionError(
                f"unexpected warmup input shape {tuple(warmup_input_ids.shape)}"
            )

        result["status"] = "warming"
        result["warmups"] = []
        result["trials"] = []
        atomic_json(args.output, result)
        for index in range(args.warmup_requests):
            record = run_generation(
                torch_module=torch,
                model=model,
                tokenizer=tokenizer,
                input_ids=warmup_input_ids,
                attention_mask=warmup_attention_mask,
                prompt_length=args.prompt_length,
                output_tokens=args.output_tokens,
                instrumentation=instrumentation,
                trial_index=index,
                warmup=True,
            )
            result["warmups"].append(record)
            atomic_json(args.output, result)

        result["status"] = "measuring"
        atomic_json(args.output, result)
        for index in range(args.repetitions):
            record = run_generation(
                torch_module=torch,
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_length=args.prompt_length,
                output_tokens=args.output_tokens,
                instrumentation=instrumentation,
                trial_index=index,
                warmup=False,
            )
            result["trials"].append(record)
            atomic_json(args.output, result)

        checks, failures = qualification_checks(
            args.mode,
            mode_config,
            result["warmups"],
            result["trials"],
            instrumentation,
            bool(executor._speculative_prefetch_overlap),
            args.prompt_length,
            args.output_tokens,
            args.warmup_requests,
            args.repetitions,
            warmup_prompt_ids == prompt_ids,
        )
        result["summary"] = {
            "ttft_seconds": summarize(
                [float(trial["ttft_seconds"]) for trial in result["trials"]]
            ),
            "decode_tpot_seconds": summarize(
                [
                    float(trial["decode_tpot_seconds"])
                    for trial in result["trials"]
                ]
            ),
            "e2e_seconds": summarize(
                [float(trial["e2e_seconds"]) for trial in result["trials"]]
            ),
            "decode_tokens_per_second": summarize(
                [
                    float(trial["decode_tokens_per_second"])
                    for trial in result["trials"]
                    if trial["decode_tokens_per_second"] is not None
                ]
            ),
            "output_hashes": sorted(
                {
                    str(trial["generated_token_ids_sha256"])
                    for trial in result["trials"]
                }
            ),
        }
        result["qualification"] = checks
        result["instrumentation_samples"] = instrumentation.samples
        result["failures"] = failures
        result["status"] = "ok" if not failures else "qualification_failed"
        atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if not failures else 2
    except BaseException as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(args.output, result)
        print(result["traceback"], end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
