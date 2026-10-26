# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import weak_ref_tensors

logger = init_logger(__name__)


@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None


@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


class CUDAGraphWrapper:
    """Wraps a runnable to add CUDA graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the cudagraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for cudagraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform cudagraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: CUDAGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
        self._cudagraph_class_entries: dict[
            str, dict[BatchDescriptor, CUDAGraphEntry]
        ] = {"default": self.concrete_cudagraph_entries}
        self._active_cudagraph_class = "default"

    @staticmethod
    def _normalize_cudagraph_class(class_name: str) -> str:
        normalized = str(class_name).strip().lower()
        if not normalized or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in normalized
        ):
            raise ValueError(f"Invalid CUDA Graph class name: {class_name!r}")
        return normalized

    def begin_cudagraph_class_capture(
        self,
        class_name: str,
        *,
        replace: bool = False,
    ) -> None:
        """Select an empty per-class cache for subsequent lazy captures."""
        normalized = self._normalize_cudagraph_class(class_name)
        existing = self._cudagraph_class_entries.get(normalized)
        if existing and not replace:
            raise RuntimeError(
                f"CUDA Graph class {normalized!r} is already populated"
            )
        if existing and replace:
            for entry in existing.values():
                if entry.cudagraph is not None:
                    entry.cudagraph.reset()
        entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
        self._cudagraph_class_entries[normalized] = entries
        self.concrete_cudagraph_entries = entries
        self._active_cudagraph_class = normalized

    def register_active_cudagraph_class(self, class_name: str) -> dict[str, Any]:
        """Name the already-captured active table without copying GraphExecs."""
        normalized = self._normalize_cudagraph_class(class_name)
        entries = self.concrete_cudagraph_entries
        if not entries or any(entry.cudagraph is None for entry in entries.values()):
            return {"ok": False, "reason": "active_class_not_fully_captured"}
        existing = self._cudagraph_class_entries.get(normalized)
        if existing is not None and existing is not entries:
            return {"ok": False, "reason": "class_already_registered"}
        old_class = self._active_cudagraph_class
        self._cudagraph_class_entries[normalized] = entries
        if old_class != normalized and (
            self._cudagraph_class_entries.get(old_class) is entries
        ):
            self._cudagraph_class_entries.pop(old_class, None)
        self._active_cudagraph_class = normalized
        return {
            "ok": True,
            "reason": "active_class_registered",
            "old_class": old_class,
            "new_class": normalized,
            "entry_count": len(entries),
        }

    def register_cudagraph_class_alias(
        self,
        class_name: str,
        source_class: str,
    ) -> dict[str, Any]:
        """Reuse one fully captured table for an equal-shape plan class."""
        normalized = self._normalize_cudagraph_class(class_name)
        source = self._normalize_cudagraph_class(source_class)
        entries = self._cudagraph_class_entries.get(source)
        if entries is None:
            return {"ok": False, "reason": "source_class_not_registered"}
        if not entries or any(entry.cudagraph is None for entry in entries.values()):
            return {"ok": False, "reason": "source_class_not_fully_captured"}
        existing = self._cudagraph_class_entries.get(normalized)
        if existing is not None and existing is not entries:
            return {"ok": False, "reason": "class_already_registered"}
        old_class = self._active_cudagraph_class
        self._cudagraph_class_entries[normalized] = entries
        self.concrete_cudagraph_entries = entries
        self._active_cudagraph_class = normalized
        return {
            "ok": True,
            "reason": "equal_shape_class_aliased",
            "old_class": old_class,
            "new_class": normalized,
            "source_class": source,
            "entry_count": len(entries),
        }

    def reset_all_cudagraph_classes(self) -> int:
        """Reset every retained GraphExec exactly once during final teardown."""
        reset_count = 0
        seen_entries: set[int] = set()
        for entries in tuple(self._cudagraph_class_entries.values()):
            if id(entries) in seen_entries:
                continue
            seen_entries.add(id(entries))
            for entry in tuple(entries.values()):
                if entry.cudagraph is not None:
                    entry.cudagraph.reset()
                    reset_count += 1
                entry.cudagraph = None
                entry.output = None
                entry.input_addresses = None
            entries.clear()
        self._cudagraph_class_entries = {"default": {}}
        self.concrete_cudagraph_entries = self._cudagraph_class_entries["default"]
        self._active_cudagraph_class = "default"
        return reset_count

    def activate_cudagraph_class(self, class_name: str) -> dict[str, Any]:
        """Atomically select an already captured class without recapture."""
        normalized = self._normalize_cudagraph_class(class_name)
        entries = self._cudagraph_class_entries.get(normalized)
        if entries is None:
            return {"ok": False, "reason": "class_not_registered"}
        uncaptured = [
            descriptor
            for descriptor, entry in entries.items()
            if entry.cudagraph is None
        ]
        if not entries or uncaptured:
            return {
                "ok": False,
                "reason": "class_not_fully_captured",
                "entry_count": len(entries),
                "uncaptured_count": len(uncaptured),
            }
        old_class = self._active_cudagraph_class
        self.concrete_cudagraph_entries = entries
        self._active_cudagraph_class = normalized
        return {
            "ok": True,
            "reason": "class_activated",
            "old_class": old_class,
            "new_class": normalized,
            "entry_count": len(entries),
        }

    def discard_cudagraph_class(
        self,
        class_name: str,
        *,
        activate_class: str = "default",
    ) -> dict[str, Any]:
        """Destroy one temporary Graph class while retaining its shared pool.

        CUDA Graphs captured by this wrapper all use ``self.graph_pool``.  A
        startup pool-seeding graph can therefore be reset after its blocks
        have established the pool layout; another graph captured into the
        same live pool can reuse those blocks.  At least one other class must
        remain live so callers cannot accidentally retire the shared pool.
        """
        normalized = self._normalize_cudagraph_class(class_name)
        target = self._normalize_cudagraph_class(activate_class)
        if normalized == target:
            return {"ok": False, "reason": "discard_target_is_activate_target"}
        entries = self._cudagraph_class_entries.get(normalized)
        target_entries = self._cudagraph_class_entries.get(target)
        if entries is None:
            return {"ok": False, "reason": "class_not_registered"}
        if not target_entries or any(
            entry.cudagraph is None for entry in target_entries.values()
        ):
            return {"ok": False, "reason": "activate_target_not_fully_captured"}

        reset_count = 0
        for entry in tuple(entries.values()):
            if entry.cudagraph is not None:
                entry.cudagraph.reset()
                reset_count += 1
            entry.cudagraph = None
            entry.output = None
            entry.input_addresses = None
        entries.clear()
        self._cudagraph_class_entries.pop(normalized, None)
        self.concrete_cudagraph_entries = target_entries
        old_class = self._active_cudagraph_class
        self._active_cudagraph_class = target
        return {
            "ok": True,
            "reason": "class_discarded_pool_retained",
            "discarded_class": normalized,
            "old_class": old_class,
            "active_class": target,
            "reset_count": reset_count,
            "active_entry_count": len(target_entries),
        }

    def cudagraph_class_diagnostics(self) -> dict[str, Any]:
        classes: dict[str, Any] = {}
        for class_name, entries in sorted(self._cudagraph_class_entries.items()):
            captured = sum(
                entry.cudagraph is not None for entry in entries.values()
            )
            classes[class_name] = {
                "entry_count": len(entries),
                "captured_count": captured,
                "fully_captured": bool(entries) and captured == len(entries),
            }
        return {
            "active_class": self._active_cudagraph_class,
            "class_count": len(classes),
            "classes": classes,
        }

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not exists in the runnable of "
            f"cudagraph wrapper: {self.runnable}"
        )

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())
                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        return entry.output
