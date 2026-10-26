# SPDX-License-Identifier: Apache-2.0
"""Application-layer UVM policy runtime facade.

This module is intentionally thin. It centralizes high-level residency actions
while using explicit CUDA/UVM range helpers. The driver remains the migration
backend; policy decisions stay at vLLM object boundaries.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from vllm.device_allocator.uvm import (
    advise_range_preferred_location,
    prefetch_range_to_device,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class UvmPolicyActionResult:
    enabled: bool
    allowed: bool
    issued: bool
    mode: str
    reason: str
    bytes: int
    range_count: int = 0


class UvmPolicyRuntime:
    """Facade for object-level UVM residency actions."""

    def __init__(self) -> None:
        self.enabled = _read_bool_env("VLLM_UVM_POLICY_RUNTIME_ENABLE", False)

    @property
    def active(self) -> bool:
        return self.enabled

    def protect_kv_region(
        self,
        *,
        region_id: str,
        block_id: int,
        group_id: int,
        bytes: int,
        ranges: list[tuple[int, int]] | None,
        mode: str,
        device: int,
        scope_key: str,
        active_region_ids: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UvmPolicyActionResult:
        """Protect an active KV region and optionally prefetch it to GPU.

        `mode` may be one of:
        - trace_only: record residency intent only.
        - advise: set preferred location to GPU.
        - prefetch: prefetch active KV ranges to GPU.
        - advise_prefetch: apply advise then prefetch.
        """
        size_bytes = max(int(bytes or 0), 0)
        ranges = list(ranges or [])

        if mode == "trace_only":
            return UvmPolicyActionResult(
                enabled=self.active,
                allowed=True,
                issued=True,
                mode=mode,
                reason="traced",
                bytes=size_bytes,
                range_count=len(ranges),
            )

        if not ranges:
            return UvmPolicyActionResult(
                enabled=self.active,
                allowed=True,
                issued=False,
                mode=mode,
                reason="missing_pointer_ranges",
                bytes=size_bytes,
                range_count=0,
            )

        try:
            advised = True
            prefetched = True
            if mode in ("advise", "advise_prefetch"):
                advised = all(
                    advise_range_preferred_location(ptr, size, device)
                    for ptr, size in ranges
                )
            if mode in ("prefetch", "advise_prefetch"):
                prefetched = all(
                    prefetch_range_to_device(ptr, size, device)
                    for ptr, size in ranges
                )
            issued = advised and prefetched
        except Exception:  # pragma: no cover - defensive UVM path.
            logger.debug("KV active protection failed for %s", region_id,
                         exc_info=True)
            issued = False

        return UvmPolicyActionResult(
            enabled=self.active,
            allowed=True,
            issued=issued,
            mode=mode,
            reason="issued" if issued else "uvm_action_failed",
            bytes=size_bytes,
            range_count=len(ranges),
        )


_POLICY_RUNTIME: UvmPolicyRuntime | None = None
_POLICY_RUNTIME_LOCK = threading.Lock()


def get_uvm_policy_runtime() -> UvmPolicyRuntime:
    global _POLICY_RUNTIME
    if _POLICY_RUNTIME is None:
        with _POLICY_RUNTIME_LOCK:
            if _POLICY_RUNTIME is None:
                _POLICY_RUNTIME = UvmPolicyRuntime()
    return _POLICY_RUNTIME
