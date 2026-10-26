# SPDX-License-Identifier: Apache-2.0
"""Application-side explicit UVM Region registration and placement."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from typing import Any

from vllm.device_allocator.uvm import (
    UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
    batch_prefetch_ranges_to_device_with_executor,
    close_uvm_slot_direct_session,
    driver_batch_query_ranges_residency_bytes,
    driver_prepare_regions_remote_read_audited,
    driver_register_region,
    driver_release_regions,
    driver_slot_execute_plan,
    driver_slot_destroy_plan,
    driver_slot_preallocate_ranges,
    driver_slot_register_plan,
    driver_slot_release_ranges,
    driver_slot_restore_ranges,
    driver_slot_transition_ranges,
    driver_unregister_region,
    driver_update_region_state,
    register_uvm_slot_direct_policy_cleanup,
)
from vllm.device_allocator.uvm_region import (
    ExplicitRegionRegistry,
    KernelRegion,
    RegionState,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


VALID_STATES = {
    "unknown",
    "gpu",
    "cpu",
    "prefetching_gpu",
    "prefetching_cpu",
    "released",
}

DRIVER_REGION_STATES = {
    "unknown": 0,
    "cpu": 1,
    "released": 1,
    "prefetching_cpu": 2,
    "gpu": 3,
    "prefetching_gpu": 4,
}

_EXPLICIT_REGION_REGISTRY = ExplicitRegionRegistry()
_EXPLICIT_REGION_LOCK = threading.Lock()

# Slot-backed PTEs deliberately do not alter ordinary UVM residency: source
# pages remain CPU-resident while the GPU reads driver-owned VIDMEM through the
# original virtual address. Keep that logical state separate from
# ``KernelRegion.state`` so callers never mistake a slot mapping for normal UVM
# GPU residency.
_SLOT_DIRECT_LOCK = threading.RLock()
_SLOT_DIRECT_OPERATION_LOCK = threading.RLock()
_SLOT_DIRECT_BINDINGS: dict[tuple[int, int, str], int] = {}
_SLOT_DIRECT_DRIVER_LAYERS: dict[tuple[int, int, str], int] = {}
_SLOT_DIRECT_SLOT_OWNERS: dict[tuple[int, int], tuple[int, int, str]] = {}
_SLOT_DIRECT_DRIVER_OWNER_TOKENS: dict[tuple[int, int], int] = {}
_SLOT_DIRECT_DRIVER_TOKEN_OWNERS: dict[int, tuple[int, int]] = {}
_SLOT_DIRECT_PREALLOCATED: set[tuple[int, int, str]] = set()
_SLOT_DIRECT_READY: set[tuple[int, int, str]] = set()
_SLOT_DIRECT_ALLOCATION_BYTES: dict[tuple[int, int, str], int] = {}
_SLOT_DIRECT_GROUP_LAST_USED: dict[tuple[int, int], int] = {}
_SLOT_DIRECT_PROTECTED_GROUPS: dict[tuple[int, int], int] = {}
_SLOT_DIRECT_USE_EPOCH = 0
_SLOT_DIRECT_ACTIVE_DEVICE: int | None = None
_SLOT_DIRECT_OWNER_THREAD: int | None = None
_SLOT_DIRECT_POISON_REASON: str | None = None
_SLOT_DIRECT_CACHE_LIMITS: tuple[int, int] | None = None
_SLOT_DIRECT_REUSE_BANK_ACTIVE: bool | None = None
# A fixed-bank lane is immutable for the lifetime of the low-level slot
# session: only its logical owner changes.  Keep the physical identity and
# allocation size so complete, same-shape layer transitions can rebind the bank
# with one linear owner lookup instead of rediscovering the 1,152 lanes on every
# layer.
_SLOT_DIRECT_FIXED_BANK_LANES: tuple[tuple[int, int, int], ...] | None = None
_SLOT_DIRECT_FIXED_BANK_PLAN_HITS = 0
_SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS = 0
_SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS = 0
_SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE: bool | None = None
_SLOT_DIRECT_LAYER_TRANSITION_ACTIVE: bool | None = None
_SLOT_DIRECT_REGISTERED_PLAN_ACTIVE: bool | None = None
_SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE: bool | None = None
_SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE: bool | None = None
_SLOT_DIRECT_REGISTERED_PLANS: dict[int, dict[str, Any]] = {}
_SLOT_DIRECT_REGISTERED_BANK_LANES: tuple[
    tuple[tuple[int, int, int], ...], ...
] | None = None
_SLOT_DIRECT_REGISTERED_BANK_KEYS: tuple[
    frozenset[tuple[int, int, str]], ...
] | None = None
_SLOT_DIRECT_INITIAL_PLAN_ID: int | None = None
_SLOT_DIRECT_PLAN_GENERATION: int | None = None
_SLOT_DIRECT_PLAN_ACTIVE_LAYER: int | None = None
# Canonical O(1) runtime state for an immutable registered dual bank. Per-slot
# policy dictionaries remain the registration snapshot and are not rewritten
# on every successful transition when this state is active.
_SLOT_DIRECT_REGISTERED_RUNTIME: dict[str, Any] | None = None
_SLOT_DIRECT_SESSION_CONTRACT: dict[str, Any] | None = None
_SLOT_DIRECT_NO_EVICT_FLAG = 0x1
_SLOT_DIRECT_DEFAULT_CACHE_BYTES = 512 * 1024 * 1024
_SLOT_DIRECT_DEFAULT_MAX_SLOTS = 8192
_SLOT_DIRECT_REUSE_BANK_ENV = "VLLM_UVM_SLOT_DIRECT_REUSE_BANK"
_SLOT_DIRECT_PRECOMPUTED_LANES_ENV = "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES"
_SLOT_DIRECT_LAYER_TRANSITION_ENV = "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION"
_SLOT_DIRECT_REGISTERED_PLAN_ENV = "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN"
_SLOT_DIRECT_DUAL_BANK_EVENT_ENV = "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT"
_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV = "VLLM_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT"
_SLOT_DIRECT_TRACE_MODE_ENV = "VLLM_UVM_SLOT_DIRECT_TRACE_MODE"
_SLOT_DIRECT_REUSE_BANK_LAYER = (1 << 32) - 1
_SLOT_DIRECT_SECOND_BANK_LAYER = (1 << 32) - 2
_SLOT_DIRECT_BUDGET_PARAMETER = (
    "/sys/module/nvidia_uvm/parameters/"
    "uvm_moe_slot_pool_vidmem_budget_bytes"
)
_SLOT_DIRECT_SESSION_ENV_DEFAULTS = {
    "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE": "0",
    "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD": "0",
    "VLLM_UVM_SLOT_DIRECT_INIT_BANK": "0",
    "VLLM_UVM_SLOT_DIRECT_REUSE_BANK": "0",
    "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES": "1",
    "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION": "0",
    "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH": "0",
    "VLLM_UVM_SLOT_DIRECT_64K_PTE": "0",
    "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN": "0",
    "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT": "0",
    "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE": "0",
    "VLLM_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT": "0",
}


def _normalize_state(state: str) -> str:
    state = (state or "unknown").strip().lower()
    return state if state in VALID_STATES else "unknown"


def _stable_driver_region_id(region_id: str) -> int:
    text = str(region_id or "")
    if not text:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    # Keep the key positive for radix-tree users and reserve 0 as invalid.
    return (int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF) or 1


def _driver_region_state_id(state: str) -> int:
    return DRIVER_REGION_STATES.get(_normalize_state(state), 0)


def _slot_direct_binding_key(
    layer_id: int,
    expert_id: int,
    region_id: str,
) -> tuple[int, int, str]:
    return int(layer_id), int(expert_id), str(region_id)


def _slot_direct_slot_id_locked(key: tuple[int, int, str]) -> int:
    reuse_bank = _slot_direct_reuse_bank_enabled()
    current = _SLOT_DIRECT_BINDINGS.get(key)
    if current is not None:
        return current

    layer_id = (
        _SLOT_DIRECT_REUSE_BANK_LAYER
        if reuse_bank
        else int(key[0])
    )
    digest = hashlib.blake2s(
        f"{key[0]}:{key[1]}:{key[2]}".encode("utf-8"),
        digest_size=4,
    ).digest()
    candidate = int.from_bytes(digest, "little") or 1
    first = candidate
    while True:
        owner = _SLOT_DIRECT_SLOT_OWNERS.get((layer_id, candidate))
        if owner is None or owner == key:
            _SLOT_DIRECT_BINDINGS[key] = candidate
            _SLOT_DIRECT_DRIVER_LAYERS[key] = layer_id
            _SLOT_DIRECT_SLOT_OWNERS[(layer_id, candidate)] = key
            return candidate
        candidate = (candidate + 1) & 0xFFFFFFFF
        if candidate == 0:
            candidate = 1
        if candidate == first:
            raise RuntimeError(f"No free slot id remains for layer {layer_id}")


def _slot_direct_reuse_bank_enabled() -> bool:
    global _SLOT_DIRECT_REUSE_BANK_ACTIVE

    value = os.environ.get(_SLOT_DIRECT_REUSE_BANK_ENV, "0").strip()
    if value not in ("0", "1"):
        raise RuntimeError(f"{_SLOT_DIRECT_REUSE_BANK_ENV} must be 0 or 1")
    requested = value == "1"
    if _SLOT_DIRECT_REUSE_BANK_ACTIVE is None:
        _SLOT_DIRECT_REUSE_BANK_ACTIVE = requested
    elif _SLOT_DIRECT_REUSE_BANK_ACTIVE != requested:
        raise RuntimeError(
            f"{_SLOT_DIRECT_REUSE_BANK_ENV} cannot change during a slot session"
        )
    return _SLOT_DIRECT_REUSE_BANK_ACTIVE


def _slot_direct_precomputed_lanes_enabled() -> bool:
    """Lock the lane-plan ablation choice for one slot session."""
    global _SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE

    value = os.environ.get(_SLOT_DIRECT_PRECOMPUTED_LANES_ENV, "1").strip()
    if value not in ("0", "1"):
        raise RuntimeError(f"{_SLOT_DIRECT_PRECOMPUTED_LANES_ENV} must be 0 or 1")
    requested = value == "1"
    if _SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE is None:
        _SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE = requested
    elif _SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE != requested:
        raise RuntimeError(
            f"{_SLOT_DIRECT_PRECOMPUTED_LANES_ENV} cannot change during a slot session"
        )
    return _SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE


def _slot_direct_layer_transition_enabled() -> bool:
    """Lock the fused owner-rebind/copy-map ablation for one slot session."""
    global _SLOT_DIRECT_LAYER_TRANSITION_ACTIVE

    value = os.environ.get(_SLOT_DIRECT_LAYER_TRANSITION_ENV, "0").strip()
    if value not in ("0", "1"):
        raise RuntimeError(f"{_SLOT_DIRECT_LAYER_TRANSITION_ENV} must be 0 or 1")
    requested = value == "1"
    if _SLOT_DIRECT_LAYER_TRANSITION_ACTIVE is None:
        _SLOT_DIRECT_LAYER_TRANSITION_ACTIVE = requested
    elif _SLOT_DIRECT_LAYER_TRANSITION_ACTIVE != requested:
        raise RuntimeError(
            f"{_SLOT_DIRECT_LAYER_TRANSITION_ENV} cannot change during a slot session"
        )
    return _SLOT_DIRECT_LAYER_TRANSITION_ACTIVE


def _slot_direct_registered_plan_enabled() -> bool:
    """Lock compact kernel-plan execution for one slot session."""
    global _SLOT_DIRECT_REGISTERED_PLAN_ACTIVE

    value = os.environ.get(_SLOT_DIRECT_REGISTERED_PLAN_ENV, "0").strip()
    if value not in ("0", "1"):
        raise RuntimeError(f"{_SLOT_DIRECT_REGISTERED_PLAN_ENV} must be 0 or 1")
    requested = value == "1"
    if _SLOT_DIRECT_REGISTERED_PLAN_ACTIVE is None:
        _SLOT_DIRECT_REGISTERED_PLAN_ACTIVE = requested
    elif _SLOT_DIRECT_REGISTERED_PLAN_ACTIVE != requested:
        raise RuntimeError(
            f"{_SLOT_DIRECT_REGISTERED_PLAN_ENV} cannot change during a slot session"
        )
    return _SLOT_DIRECT_REGISTERED_PLAN_ACTIVE


def _slot_direct_dual_bank_event_enabled() -> bool:
    """Freeze the experimental two-backing-bank registered-plan contract."""
    global _SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE

    value = os.environ.get(_SLOT_DIRECT_DUAL_BANK_EVENT_ENV, "0").strip()
    if value not in ("0", "1"):
        raise RuntimeError(f"{_SLOT_DIRECT_DUAL_BANK_EVENT_ENV} must be 0 or 1")
    requested = value == "1"
    if _SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE is None:
        _SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE = requested
    elif _SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE != requested:
        raise RuntimeError(
            f"{_SLOT_DIRECT_DUAL_BANK_EVENT_ENV} cannot change during a slot session"
        )
    return _SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE


def _slot_direct_bank_level_commit_enabled() -> bool:
    """Freeze O(1) userspace commit for one registered-plan session."""
    global _SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE

    value = os.environ.get(_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV, "0").strip()
    if value not in ("0", "1"):
        raise RuntimeError(f"{_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV} must be 0 or 1")
    requested = value == "1"
    if _SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE is None:
        _SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE = requested
    elif _SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE != requested:
        raise RuntimeError(
            f"{_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV} cannot change during a slot session"
        )
    return _SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE


def _slot_direct_session_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    for name, default in _SLOT_DIRECT_SESSION_ENV_DEFAULTS.items():
        value = os.environ.get(name, default).strip()
        if value not in ("0", "1"):
            raise RuntimeError(f"{name} must be 0 or 1")
        values[name] = value
    access_mode = os.environ.get(
        "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "restore"
    ).strip().lower()
    if not access_mode:
        raise RuntimeError("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE must not be empty")
    values["VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE"] = access_mode
    for name in (
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES",
        "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            _slot_direct_positive_limit(name, value)
        values[name] = value
    return values


def freeze_uvm_slot_direct_session_contract(
    *,
    registered_plan: bool,
    access_mode: str,
    device: int,
) -> dict[str, Any]:
    """Freeze every environment choice which controls one slot session."""
    global _SLOT_DIRECT_SESSION_CONTRACT

    normalized_access = str(access_mode).strip().lower()
    environment = _slot_direct_session_environment()
    requested_registered = bool(registered_plan)
    if environment["VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE"] != normalized_access:
        raise RuntimeError("UVM slot session access mode changed before freeze")
    if (
        environment["VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN"] == "1"
    ) != requested_registered:
        raise RuntimeError("UVM registered-plan mode changed before freeze")
    candidate: dict[str, Any] = {
        "schema_version": 1,
        "device": int(device),
        "access_mode": normalized_access,
        "registered_plan": requested_registered,
        "slot_session_enabled": normalized_access == "slot_restore",
        "environment": tuple(sorted(environment.items())),
    }
    with _SLOT_DIRECT_LOCK:
        if _SLOT_DIRECT_SESSION_CONTRACT is None:
            _SLOT_DIRECT_SESSION_CONTRACT = candidate
        elif _SLOT_DIRECT_SESSION_CONTRACT != candidate:
            raise RuntimeError("UVM slot session contract changed after initialization")
        return dict(_SLOT_DIRECT_SESSION_CONTRACT)


def validate_uvm_slot_direct_session_contract(
    *,
    access_mode: str,
    device: int,
) -> dict[str, Any]:
    """Return the frozen contract only if caller and environment still match."""
    with _SLOT_DIRECT_LOCK:
        contract = (
            None
            if _SLOT_DIRECT_SESSION_CONTRACT is None
            else dict(_SLOT_DIRECT_SESSION_CONTRACT)
        )
    if contract is None:
        raise RuntimeError("UVM slot session contract was not initialized")
    if int(contract["device"]) != int(device):
        raise RuntimeError("UVM slot session device changed after initialization")
    if str(contract["access_mode"]) != str(access_mode).strip().lower():
        raise RuntimeError("UVM slot session access mode changed after initialization")
    current = tuple(sorted(_slot_direct_session_environment().items()))
    if current != contract["environment"]:
        raise RuntimeError("UVM slot session environment changed after initialization")
    return contract


def get_uvm_slot_direct_session_contract() -> dict[str, Any] | None:
    """Read frozen shutdown state without consulting mutable environment."""
    with _SLOT_DIRECT_LOCK:
        return (
            None
            if _SLOT_DIRECT_SESSION_CONTRACT is None
            else dict(_SLOT_DIRECT_SESSION_CONTRACT)
        )


def _slot_direct_driver_identity_locked(
    key: tuple[int, int, str],
) -> tuple[int, int]:
    return (
        int(_SLOT_DIRECT_DRIVER_LAYERS.get(key, key[0])),
        int(_SLOT_DIRECT_BINDINGS[key]),
    )


def _slot_direct_driver_owner_token_locked(
    layer_id: int,
    expert_id: int,
) -> int:
    """Return a collision-free driver owner token for one logical expert."""
    if not _slot_direct_reuse_bank_enabled():
        return int(expert_id)

    group = (int(layer_id), int(expert_id))
    current = _SLOT_DIRECT_DRIVER_OWNER_TOKENS.get(group)
    if current is not None:
        return current

    digest = hashlib.blake2s(
        f"owner:{group[0]}:{group[1]}".encode("utf-8"),
        digest_size=4,
    ).digest()
    candidate = int.from_bytes(digest, "little") or 1
    first = candidate
    while True:
        owner = _SLOT_DIRECT_DRIVER_TOKEN_OWNERS.get(candidate)
        if owner is None or owner == group:
            _SLOT_DIRECT_DRIVER_OWNER_TOKENS[group] = candidate
            _SLOT_DIRECT_DRIVER_TOKEN_OWNERS[candidate] = group
            return candidate
        candidate = (candidate + 1) & 0xFFFFFFFF
        if candidate == 0:
            candidate = 1
        if candidate == first:
            raise RuntimeError("No free driver slot owner token remains")


def _slot_direct_capture_fixed_bank_lanes_locked(
    keys: Sequence[tuple[int, int, str]],
) -> bool:
    """Capture one complete allocated bank in immutable physical-lane order."""
    global _SLOT_DIRECT_FIXED_BANK_LANES
    global _SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS
    global _SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS

    if (
        not _slot_direct_reuse_bank_enabled()
        or not _slot_direct_precomputed_lanes_enabled()
        or _SLOT_DIRECT_FIXED_BANK_LANES is not None
    ):
        return False
    key_set = set(keys)
    if not key_set or key_set != _SLOT_DIRECT_PREALLOCATED:
        return False

    started_ns = time.perf_counter_ns()
    ordered_keys = sorted(
        key_set,
        key=lambda key: (_SLOT_DIRECT_ALLOCATION_BYTES.get(key, -1), key),
    )
    lanes: list[tuple[int, int, int]] = []
    identities: set[tuple[int, int]] = set()
    for key in ordered_keys:
        allocation_bytes = _SLOT_DIRECT_ALLOCATION_BYTES.get(key)
        if allocation_bytes is None:
            return False
        driver_layer, slot_id = _slot_direct_driver_identity_locked(key)
        identity = (int(driver_layer), int(slot_id))
        if (
            driver_layer != _SLOT_DIRECT_REUSE_BANK_LAYER
            or identity in identities
            or _SLOT_DIRECT_SLOT_OWNERS.get(identity) != key
        ):
            return False
        identities.add(identity)
        lanes.append(
            (
                identity[0],
                identity[1],
                int(allocation_bytes),
            )
        )

    _SLOT_DIRECT_FIXED_BANK_LANES = tuple(lanes)
    _SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS += 1
    _SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS += time.perf_counter_ns() - started_ns
    return True


def _slot_direct_fixed_bank_plan_stats_locked() -> tuple[int, int, int]:
    return (
        int(_SLOT_DIRECT_FIXED_BANK_PLAN_HITS),
        int(_SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS),
        int(_SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS),
    )


def _slot_direct_preallocate_only_preflight_locked(
    keys: Sequence[tuple[int, int, str]],
    records: dict[str, KernelRegion | None],
    *,
    device: int,
) -> tuple[str, str]:
    """Classify init-bank as pristine, exactly idempotent, or conflicting."""
    key_set = set(keys)
    if len(key_set) != len(keys):
        return "conflict", "initial bank contains duplicate logical lanes"

    has_bank_state = bool(
        _SLOT_DIRECT_BINDINGS
        or _SLOT_DIRECT_DRIVER_LAYERS
        or _SLOT_DIRECT_SLOT_OWNERS
        or _SLOT_DIRECT_DRIVER_OWNER_TOKENS
        or _SLOT_DIRECT_DRIVER_TOKEN_OWNERS
        or _SLOT_DIRECT_PREALLOCATED
        or _SLOT_DIRECT_READY
        or _SLOT_DIRECT_ALLOCATION_BYTES
        or _SLOT_DIRECT_GROUP_LAST_USED
        or _SLOT_DIRECT_PROTECTED_GROUPS
        or _SLOT_DIRECT_FIXED_BANK_LANES is not None
        or _SLOT_DIRECT_REGISTERED_BANK_LANES is not None
        or _SLOT_DIRECT_REGISTERED_BANK_KEYS is not None
        or _SLOT_DIRECT_REGISTERED_PLANS
        or _SLOT_DIRECT_INITIAL_PLAN_ID is not None
        or _SLOT_DIRECT_PLAN_GENERATION is not None
        or _SLOT_DIRECT_PLAN_ACTIVE_LAYER is not None
        or _SLOT_DIRECT_ACTIVE_DEVICE is not None
        or _SLOT_DIRECT_OWNER_THREAD is not None
        or _SLOT_DIRECT_POISON_REASON is not None
    )
    if not has_bank_state:
        return "initialize", ""

    if _SLOT_DIRECT_POISON_REASON is not None:
        return "conflict", "slot session is poisoned"
    if _SLOT_DIRECT_ACTIVE_DEVICE != int(device):
        return "conflict", "slot session is bound to another device"
    if (
        _SLOT_DIRECT_OWNER_THREAD is not None
        and _SLOT_DIRECT_OWNER_THREAD != threading.get_ident()
    ):
        return "conflict", "slot session is owned by another thread"
    if _SLOT_DIRECT_READY:
        return "conflict", "initial bank already has GPU-visible mappings"
    if _SLOT_DIRECT_GROUP_LAST_USED or _SLOT_DIRECT_PROTECTED_GROUPS:
        return "conflict", "initial bank already has runtime usage state"
    if (
        _SLOT_DIRECT_REGISTERED_PLANS
        or _SLOT_DIRECT_INITIAL_PLAN_ID is not None
        or _SLOT_DIRECT_PLAN_GENERATION is not None
        or _SLOT_DIRECT_PLAN_ACTIVE_LAYER is not None
    ):
        return "conflict", "registered plans already exist"
    if _SLOT_DIRECT_PREALLOCATED != key_set:
        return "conflict", "initial bank logical lane set changed"
    if set(_SLOT_DIRECT_BINDINGS) != key_set:
        return "conflict", "initial bank binding set changed"
    if set(_SLOT_DIRECT_DRIVER_LAYERS) != key_set:
        return "conflict", "initial bank driver-layer set changed"
    if set(_SLOT_DIRECT_ALLOCATION_BYTES) != key_set:
        return "conflict", "initial bank allocation set changed"
    if not _SLOT_DIRECT_FIXED_BANK_LANES:
        return "conflict", "initial bank lane plan is missing"
    identities = [
        (
            int(_SLOT_DIRECT_DRIVER_LAYERS[key]),
            int(_SLOT_DIRECT_BINDINGS[key]),
        )
        for key in keys
    ]
    if (
        len(set(identities)) != len(key_set)
        or len(_SLOT_DIRECT_SLOT_OWNERS) != len(key_set)
        or len(_SLOT_DIRECT_FIXED_BANK_LANES) != len(key_set)
    ):
        return "conflict", "initial bank physical lane mapping is not bijective"

    expected_groups = {(int(key[0]), int(key[1])) for key in keys}
    if set(_SLOT_DIRECT_DRIVER_OWNER_TOKENS) != expected_groups:
        return "conflict", "initial bank expert owner-token set changed"
    if (
        len(_SLOT_DIRECT_DRIVER_TOKEN_OWNERS) != len(expected_groups)
        or set(_SLOT_DIRECT_DRIVER_TOKEN_OWNERS.values()) != expected_groups
        or any(
            _SLOT_DIRECT_DRIVER_TOKEN_OWNERS.get(token) != group
            for group, token in _SLOT_DIRECT_DRIVER_OWNER_TOKENS.items()
        )
    ):
        return "conflict", "initial bank reverse owner-token set changed"

    expected_lanes: set[tuple[int, int, int]] = set()
    expected_owners: dict[tuple[int, int], tuple[int, int, str]] = {}
    physical_identities: set[tuple[int, int]] = set()
    for key in keys:
        record = records.get(key[2])
        if record is None:
            return "conflict", f"initial bank region disappeared: {key[2]}"
        if _SLOT_DIRECT_ALLOCATION_BYTES.get(key) != int(record.length):
            return "conflict", f"initial bank lane size changed: {key[2]}"
        slot_id = _SLOT_DIRECT_BINDINGS[key]
        driver_layer = _SLOT_DIRECT_DRIVER_LAYERS[key]
        identity = (int(driver_layer), int(slot_id))
        if identity in physical_identities:
            return "conflict", "initial bank repeats a physical lane identity"
        physical_identities.add(identity)
        expected_owners[identity] = key
        expected_lanes.add((identity[0], identity[1], int(record.length)))
    if (
        len(physical_identities) != len(keys)
        or len(expected_owners) != len(keys)
        or len(expected_lanes) != len(keys)
    ):
        return "conflict", "initial bank physical lane bijection changed"
    if _SLOT_DIRECT_SLOT_OWNERS != expected_owners:
        return "conflict", "initial bank physical owner table changed"
    if (
        len(_SLOT_DIRECT_FIXED_BANK_LANES) != len(keys)
        or len(set(_SLOT_DIRECT_FIXED_BANK_LANES)) != len(keys)
        or set(_SLOT_DIRECT_FIXED_BANK_LANES) != expected_lanes
    ):
        return "conflict", "initial bank physical lane geometry changed"
    return "idempotent", ""


def _slot_direct_positive_limit(name: str, value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return parsed


def _slot_direct_cache_limits_locked() -> tuple[int, int]:
    global _SLOT_DIRECT_CACHE_LIMITS

    if _SLOT_DIRECT_CACHE_LIMITS is not None:
        return _SLOT_DIRECT_CACHE_LIMITS

    cache_text = os.environ.get("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", "").strip()
    if cache_text:
        cache_bytes = _slot_direct_positive_limit(
            "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", cache_text
        )
    else:
        try:
            with open(_SLOT_DIRECT_BUDGET_PARAMETER, encoding="utf-8") as handle:
                cache_bytes = _slot_direct_positive_limit(
                    _SLOT_DIRECT_BUDGET_PARAMETER, handle.read().strip()
                )
        except OSError:
            cache_bytes = _SLOT_DIRECT_DEFAULT_CACHE_BYTES

    slots_text = os.environ.get("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "").strip()
    max_slots = (
        _slot_direct_positive_limit(
            "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", slots_text
        )
        if slots_text
        else _SLOT_DIRECT_DEFAULT_MAX_SLOTS
    )
    _SLOT_DIRECT_CACHE_LIMITS = cache_bytes, max_slots
    return _SLOT_DIRECT_CACHE_LIMITS


def _slot_direct_cache_snapshot_locked() -> dict[str, int]:
    cache_bytes, max_slots = _slot_direct_cache_limits_locked()
    if _SLOT_DIRECT_REGISTERED_RUNTIME is not None:
        runtime = _SLOT_DIRECT_REGISTERED_RUNTIME
        return {
            "allocated_bytes": int(runtime["allocated_bytes"]),
            "allocated_slots": int(runtime["allocated_slots"]),
            "ready_slots": int(runtime["ready_slots"]),
            "resident_groups": int(runtime["resident_groups"]),
            "cache_bytes_limit": cache_bytes,
            "cache_slots_limit": max_slots,
        }
    return {
        "allocated_bytes": sum(_SLOT_DIRECT_ALLOCATION_BYTES.values()),
        "allocated_slots": len(_SLOT_DIRECT_PREALLOCATED),
        "ready_slots": len(_SLOT_DIRECT_READY),
        "resident_groups": len(
            {
                (key[0], key[1])
                for key in _SLOT_DIRECT_PREALLOCATED
            }
        ),
        "cache_bytes_limit": cache_bytes,
        "cache_slots_limit": max_slots,
    }


def _slot_direct_touch_group_locked(layer_id: int, expert_id: int) -> None:
    global _SLOT_DIRECT_USE_EPOCH

    _SLOT_DIRECT_USE_EPOCH += 1
    _SLOT_DIRECT_GROUP_LAST_USED[(int(layer_id), int(expert_id))] = (
        _SLOT_DIRECT_USE_EPOCH
    )


def _slot_direct_forget_key_locked(key: tuple[int, int, str]) -> None:
    global _SLOT_DIRECT_ACTIVE_DEVICE, _SLOT_DIRECT_FIXED_BANK_LANES

    _SLOT_DIRECT_READY.discard(key)
    _SLOT_DIRECT_PREALLOCATED.discard(key)
    _SLOT_DIRECT_ALLOCATION_BYTES.pop(key, None)
    slot_id = _SLOT_DIRECT_BINDINGS.pop(key, None)
    driver_layer = _SLOT_DIRECT_DRIVER_LAYERS.pop(key, int(key[0]))
    if slot_id is not None:
        owner_key = (int(driver_layer), int(slot_id))
        if _SLOT_DIRECT_SLOT_OWNERS.get(owner_key) == key:
            _SLOT_DIRECT_SLOT_OWNERS.pop(owner_key, None)
        # An explicit release destroys this physical lane. A later complete
        # bank may have different slot identities and must be learned anew.
        _SLOT_DIRECT_FIXED_BANK_LANES = None

    group = (int(key[0]), int(key[1]))
    if not any((value[0], value[1]) == group for value in _SLOT_DIRECT_BINDINGS):
        _SLOT_DIRECT_GROUP_LAST_USED.pop(group, None)
    if not _SLOT_DIRECT_BINDINGS:
        _SLOT_DIRECT_ACTIVE_DEVICE = None


def _clear_uvm_slot_direct_policy_state(device: int) -> None:
    """Forget policy metadata after the low-level fd/session was closed."""
    global _SLOT_DIRECT_ACTIVE_DEVICE, _SLOT_DIRECT_OWNER_THREAD
    global _SLOT_DIRECT_POISON_REASON, _SLOT_DIRECT_USE_EPOCH
    global _SLOT_DIRECT_REUSE_BANK_ACTIVE, _SLOT_DIRECT_FIXED_BANK_LANES
    global _SLOT_DIRECT_FIXED_BANK_PLAN_HITS
    global _SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS
    global _SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS
    global _SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE
    global _SLOT_DIRECT_LAYER_TRANSITION_ACTIVE
    global _SLOT_DIRECT_REGISTERED_PLAN_ACTIVE
    global _SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE
    global _SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE
    global _SLOT_DIRECT_REGISTERED_BANK_LANES
    global _SLOT_DIRECT_REGISTERED_BANK_KEYS
    global _SLOT_DIRECT_INITIAL_PLAN_ID, _SLOT_DIRECT_PLAN_GENERATION
    global _SLOT_DIRECT_PLAN_ACTIVE_LAYER
    global _SLOT_DIRECT_REGISTERED_RUNTIME
    global _SLOT_DIRECT_SESSION_CONTRACT
    global _SLOT_DIRECT_CACHE_LIMITS

    with _SLOT_DIRECT_OPERATION_LOCK, _SLOT_DIRECT_LOCK:
        if (
            _SLOT_DIRECT_ACTIVE_DEVICE is not None
            and _SLOT_DIRECT_ACTIVE_DEVICE != int(device)
        ):
            return
        _SLOT_DIRECT_BINDINGS.clear()
        _SLOT_DIRECT_DRIVER_LAYERS.clear()
        _SLOT_DIRECT_SLOT_OWNERS.clear()
        _SLOT_DIRECT_DRIVER_OWNER_TOKENS.clear()
        _SLOT_DIRECT_DRIVER_TOKEN_OWNERS.clear()
        _SLOT_DIRECT_PREALLOCATED.clear()
        _SLOT_DIRECT_READY.clear()
        _SLOT_DIRECT_ALLOCATION_BYTES.clear()
        _SLOT_DIRECT_GROUP_LAST_USED.clear()
        _SLOT_DIRECT_PROTECTED_GROUPS.clear()
        _SLOT_DIRECT_USE_EPOCH = 0
        _SLOT_DIRECT_ACTIVE_DEVICE = None
        _SLOT_DIRECT_OWNER_THREAD = None
        _SLOT_DIRECT_POISON_REASON = None
        _SLOT_DIRECT_REUSE_BANK_ACTIVE = None
        _SLOT_DIRECT_FIXED_BANK_LANES = None
        _SLOT_DIRECT_FIXED_BANK_PLAN_HITS = 0
        _SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS = 0
        _SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS = 0
        _SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE = None
        _SLOT_DIRECT_LAYER_TRANSITION_ACTIVE = None
        _SLOT_DIRECT_REGISTERED_PLAN_ACTIVE = None
        _SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE = None
        _SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE = None
        _SLOT_DIRECT_REGISTERED_PLANS.clear()
        _SLOT_DIRECT_REGISTERED_BANK_LANES = None
        _SLOT_DIRECT_REGISTERED_BANK_KEYS = None
        _SLOT_DIRECT_INITIAL_PLAN_ID = None
        _SLOT_DIRECT_PLAN_GENERATION = None
        _SLOT_DIRECT_PLAN_ACTIVE_LAYER = None
        _SLOT_DIRECT_REGISTERED_RUNTIME = None
        _SLOT_DIRECT_SESSION_CONTRACT = None
        _SLOT_DIRECT_CACHE_LIMITS = None


register_uvm_slot_direct_policy_cleanup(_clear_uvm_slot_direct_policy_state)


@contextmanager
def protect_uvm_slot_direct_groups(
    groups: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> Iterator[None]:
    """Prevent one upcoming fused-MoE working set from evicting itself."""
    normalized = sorted(
        {
            (int(layer_id), int(expert_id))
            for layer_id, expert_id in groups
        }
    )
    with _SLOT_DIRECT_LOCK:
        for group in normalized:
            _SLOT_DIRECT_PROTECTED_GROUPS[group] = (
                _SLOT_DIRECT_PROTECTED_GROUPS.get(group, 0) + 1
            )
    try:
        yield
    finally:
        with _SLOT_DIRECT_LOCK:
            for group in normalized:
                remaining = _SLOT_DIRECT_PROTECTED_GROUPS.get(group, 0) - 1
                if remaining > 0:
                    _SLOT_DIRECT_PROTECTED_GROUPS[group] = remaining
                else:
                    _SLOT_DIRECT_PROTECTED_GROUPS.pop(group, None)


def _slot_direct_trace_mode() -> str:
    mode = os.environ.get(_SLOT_DIRECT_TRACE_MODE_ENV, "full").strip().lower()
    if mode not in ("full", "compact", "errors", "off"):
        logger.warning(
            "Ignoring invalid %s=%r; using full trace",
            _SLOT_DIRECT_TRACE_MODE_ENV,
            mode,
        )
        return "full"
    return mode


def _slot_direct_compact_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded success record without per-entry arrays or maps."""
    operation = str(record.get("operation", ""))
    common_fields = (
        "operation",
        "ok",
        "reason",
        "error",
        "scope",
        "trigger",
        "wall_ns",
        "layer_id",
        "expert_id",
        "expert_count",
        "expert_ids",
        "requested_bytes",
        "logical_placement",
        "ordinary_uvm_placement",
        "registered_plan_used",
        "dual_bank_event",
        "bank_level_commit",
        "userspace_commit_slots",
        "bank_id",
        "plan_id",
        "plan_generation",
        "plan_new_generation",
        "poisoned",
        "poison_reason",
        "bank_rebind_slots",
        "cache_before",
        "cache_after",
        "limits",
    )
    compact = {
        key: record[key]
        for key in common_fields
        if key in record
    }
    compact["trace_schema"] = "slot_direct_compact_v1"

    if operation == "register_layer_plans":
        for key in (
            "initial_layer",
            "initial_plan_id",
            "generation",
            "layer_count",
            "lane_count",
            "bank_count",
            "dual_bank_event",
            "bank_level_commit",
            "requested_bytes_per_layer",
        ):
            if key in record:
                compact[key] = record[key]
    elif operation == "restore_layer":
        low_level = record.get("restore")
        if isinstance(low_level, dict):
            low_fields = (
                "ok",
                "reason",
                "plan_execute",
                "plan_id",
                "layer_id",
                "mode",
                "generation",
                "new_generation",
                "requested_count",
                "success_count",
                "failure_count",
                "conflict_count",
                "requested_bytes",
                "accepted_bytes",
                "remap_count",
                "tlb_invalidate_count",
                "copy_push_count",
                "owner_cutover",
                "owner_cutover_count",
                "owner_cutover_trusted",
                "poisoned",
                "kernel_poisoned",
                "cuda_fence_used",
                "cuda_fence_ticket",
                "cuda_compute_fence_wall_ns",
                "cuda_fence_accepted",
                "cuda_completion_signaled",
                "cuda_stream_wait_enqueued",
                "success_contract_ok",
                "failure_contract_ok",
                "preflight_rejection_ok",
                "new_generation_trusted",
                "entry_statuses_elided",
                "ioctl_ret",
                "errno",
                "rm_status",
                "ioctl_calls",
                "ioctl_wall_ns",
                "wall_ns",
            )
            compact["restore"] = {
                key: low_level[key]
                for key in low_fields
                if key in low_level
            }
    else:
        for key in (
            "requested_count",
            "success_count",
            "failure_count",
            "released_bytes",
            "released_slots",
            "closed",
        ):
            if key in record:
                compact[key] = record[key]

    return compact


def _write_slot_direct_trace(record: dict[str, Any]) -> None:
    path = os.environ.get("VLLM_UVM_SLOT_DIRECT_TRACE_FILE", "").strip()
    if not path:
        return
    mode = _slot_direct_trace_mode()
    if mode == "off" or (mode == "errors" and record.get("ok") is True):
        return
    payload_record = (
        _slot_direct_compact_trace_record(record)
        if mode == "compact" and record.get("ok") is True
        else record
    )
    payload = {"timestamp": time.time(), **payload_record}
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        logger.warning("Failed to write slot-direct trace: %s", path, exc_info=True)


def _trace_slot_direct_operation(operation: str):
    """Trace every returned failure, including validation early returns."""

    def decorate(function):
        @wraps(function)
        def traced(*args, **kwargs):
            started_ns = time.perf_counter_ns()
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                failure: dict[str, Any] = {
                    "operation": operation,
                    "ok": False,
                    "reason": "exception",
                    "error": repr(exc),
                    "wall_ns": time.perf_counter_ns() - started_ns,
                }
                if "layer_id" in kwargs:
                    failure["layer_id"] = int(kwargs["layer_id"])
                _write_slot_direct_trace(failure)
                raise

            if isinstance(result, dict):
                record = {"operation": operation, **result}
                record.setdefault("wall_ns", time.perf_counter_ns() - started_ns)
                _write_slot_direct_trace(record)
            return result

        return traced

    return decorate


def register_uvm_region(
    region_id: str,
    base: int,
    length: int,
    *,
    state: str = "unknown",
) -> KernelRegion:
    """Register geometry with an UNKNOWN initial state in both tables."""
    with _EXPLICIT_REGION_LOCK:
        region = _EXPLICIT_REGION_REGISTRY.register_region(
            region_id,
            base,
            length,
            RegionState.UNKNOWN,
        )
        driver_region_id = _stable_driver_region_id(region.region_id)
        try:
            status = driver_register_region(
                region_id=driver_region_id,
                ptr=region.base,
                nbytes=region.length,
            )
        except Exception as exc:
            region.state = RegionState.UNKNOWN
            _EXPLICIT_REGION_REGISTRY.regions.pop(region.region_id, None)
            raise RuntimeError(
                "Explicit UVM driver region registration raised an exception: "
                f"region_id={region.region_id!r}, "
                f"driver_region_id={driver_region_id}, "
                f"base=0x{region.base:x}, length={region.length}"
            ) from exc

        # Some lightweight unit-test mocks are side-effect-only and return
        # None.  The production boundary always returns an integer status.
        status_code = 0 if status is None else int(status)
        if status_code < 0:
            region.state = RegionState.UNKNOWN
            # The application table must contain only driver-registered
            # regions.  Otherwise a later raw-range Hot prefetch could hide
            # this registration failure and incorrectly commit GPU state.
            _EXPLICIT_REGION_REGISTRY.regions.pop(region.region_id, None)
            raise RuntimeError(
                "Explicit UVM driver region registration failed: "
                f"region_id={region.region_id!r}, "
                f"driver_region_id={driver_region_id}, "
                f"base=0x{region.base:x}, length={region.length}, "
                f"status={status_code}"
            )

        # Kernel registration itself initializes the state to UNKNOWN. Keep
        # registration single-phase so a redundant update cannot leave the
        # application and driver tables disagreeing about initial placement.
        return region


def unregister_uvm_regions(
    region_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Remove explicit driver/application records for named live ranges."""
    requested = list(dict.fromkeys(str(value) for value in region_ids))
    statuses: dict[str, bool] = {}
    with _EXPLICIT_REGION_LOCK:
        records = {
            region_id: _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            for region_id in requested
        }
    for region_id, region in records.items():
        if region is None:
            statuses[region_id] = False
            continue
        status = driver_unregister_region(_stable_driver_region_id(region_id))
        ok = int(status) >= 0
        statuses[region_id] = ok
        if ok:
            with _EXPLICIT_REGION_LOCK:
                current = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
                if current is region:
                    _EXPLICIT_REGION_REGISTRY.regions.pop(region_id, None)
    return {
        "ok": all(statuses.get(region_id, False) for region_id in requested),
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses.get(region_id, False)
        ],
    }


def release_uvm_regions(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    remote_read: bool = False,
    immutable_cpu_shadow: bool = False,
    include_audit: bool = False,
) -> dict[str, Any]:
    """Release only the explicitly named regions, with idempotent state checks."""
    requested = [str(region_id) for region_id in region_ids]
    statuses: dict[str, bool] = {}
    audits: dict[str, dict[str, Any]] = {}
    pending: list[KernelRegion] = []

    def audit_stub(
        region_id: str,
        *,
        region: KernelRegion | None,
        ok: bool,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "region_id": region_id,
            "driver_region_id": (
                _stable_driver_region_id(region_id) if region is not None else 0
            ),
            "registered_base": int(region.base) if region is not None else 0,
            "registered_length": int(region.length) if region is not None else 0,
            "ok": bool(ok),
            "made_progress": False,
            "reason": reason,
            "ioctl_ret": 0 if ok else -1,
            "errno": 0,
            "rm_status": 0,
            "flags": 0,
            "remote_read": bool(remote_read),
            "base": int(region.base) if region is not None else 0,
            "length": int(region.length) if region is not None else 0,
            "block_count": 0,
            "attempted_blocks": 0,
            "success_blocks": 0,
            "failure_blocks": 0,
            "skipped_blocks": 0,
            "attempted_bytes": 0,
            "released_bytes": 0,
            "failed_bytes": 0,
            "remote_map_blocks": 0,
            "remote_map_failure_blocks": 0,
            "cpu_resident_bytes": 0,
            "remote_mapped_bytes": 0,
            "newly_remote_mapped_bytes": 0,
            "first_failed_block": 0,
            "first_error": 0,
            "app_state_after": (
                region.state.value if region is not None else "missing"
            ),
        }

    with _EXPLICIT_REGION_LOCK:
        for region_id in requested:
            region = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            if region is None:
                statuses[region_id] = False
                if include_audit:
                    audits[region_id] = audit_stub(
                        region_id,
                        region=None,
                        ok=False,
                        reason="unregistered_region",
                    )
            elif region.state == RegionState.CPU and not remote_read:
                statuses[region_id] = True
                if include_audit:
                    audits[region_id] = audit_stub(
                        region_id,
                        region=region,
                        ok=True,
                        reason="already_cpu",
                    )
            else:
                # UNKNOWN is not a proof that the range is already on either
                # device. CPU placement also does not prove that remote-read
                # mappings exist. Submit the conservative operation and only
                # commit CPU after the driver reports success.
                region.state = RegionState.PREFETCHING_CPU
                pending.append(region)

    if pending:
        numeric_to_region = {
            _stable_driver_region_id(region.region_id): region for region in pending
        }
        try:
            driver_kwargs: dict[str, Any] = {
                "device": device,
                "remote_read": remote_read,
                "immutable_cpu_shadow": immutable_cpu_shadow,
            }
            if include_audit:
                driver_kwargs["include_audit"] = True
            result = driver_release_regions(list(numeric_to_region), **driver_kwargs)
        except Exception:
            logger.warning("Failed to release explicit UVM regions", exc_info=True)
            result = {"ok": False, "statuses": {}}
        driver_statuses = result.get("statuses")
        if not isinstance(driver_statuses, dict):
            driver_statuses = {}
        driver_audits = result.get("audits") if include_audit else None
        if not isinstance(driver_audits, dict):
            driver_audits = {}
        with _EXPLICIT_REGION_LOCK:
            for numeric_id, region in numeric_to_region.items():
                ok = bool(driver_statuses.get(numeric_id, False))
                region.state = RegionState.CPU if ok else RegionState.UNKNOWN
                statuses[region.region_id] = ok
                driver_update_region_state(
                    region_id=numeric_id,
                    state=_driver_region_state_id(region.state.value),
                )
                if include_audit:
                    raw_audit = driver_audits.get(
                        numeric_id,
                        driver_audits.get(str(numeric_id)),
                    )
                    if isinstance(raw_audit, dict):
                        audits[region.region_id] = {
                            **raw_audit,
                            "region_id": region.region_id,
                            "driver_region_id": int(numeric_id),
                            "registered_base": int(region.base),
                            "registered_length": int(region.length),
                            "app_state_after": region.state.value,
                        }
                    else:
                        audits[region.region_id] = audit_stub(
                            region.region_id,
                            region=region,
                            ok=False,
                            reason="driver_audit_missing",
                        )

    response: dict[str, Any] = {
        "ok": all(statuses.get(region_id, False) for region_id in requested),
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses.get(region_id, False)
        ],
    }
    if include_audit:
        response["audits"] = audits
    return response


def transition_uvm_regions_to_remote_read(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
) -> dict[str, Any]:
    """Synchronously establish audited remote-read mappings for a fixed set.

    This is intentionally stricter than a normal idempotent release.  It is
    used after a slot-direct session has been closed, where silently accepting
    CPU residency without proving the remote READ PTE would leave the next
    model forward able to fault through an unknown mapping.
    """
    requested = list(dict.fromkeys(str(region_id) for region_id in region_ids))
    batch_enabled = os.environ.get(
        "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH", "1"
    ).strip().lower() not in ("0", "false", "no", "off")
    if batch_enabled:
        registered_regions: list[tuple[int, int, int]] = []
        statuses: dict[str, bool] = {}
        with _EXPLICIT_REGION_LOCK:
            snapshots: dict[str, tuple[KernelRegion, int, int]] = {}
            for region_id in requested:
                region = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
                valid = (
                    region is not None
                    and region.state == RegionState.CPU
                    and int(region.base) % (64 * 1024) == 0
                    and int(region.length) % (64 * 1024) == 0
                )
                statuses[region_id] = valid
                if valid:
                    assert region is not None
                    snapshots[region_id] = (
                        region,
                        int(region.base),
                        int(region.length),
                    )
                    registered_regions.append(
                        (
                            _stable_driver_region_id(region_id),
                            int(region.base),
                            int(region.length),
                        )
                    )
            if not all(statuses.values()) or len(registered_regions) != len(requested):
                return {
                    "ok": False,
                    "reason": "remote_read_batch_preflight_failed",
                    "region_count": len(requested),
                    "statuses": statuses,
                    "failed_region_ids": [
                        region_id
                        for region_id in requested
                        if not statuses.get(region_id, False)
                    ],
                    "batch": True,
                }

            # Keep registry identity and geometry frozen across all chunks.
            # The runner separately prevents model work during this one-shot
            # P3-close-to-remote cutover.
            for region_id in requested:
                snapshots[region_id][0].state = RegionState.PREFETCHING_CPU
            batch_result = driver_prepare_regions_remote_read_audited(
                registered_regions,
                device=int(device),
            )
            ok = bool(batch_result.get("ok"))
            identity_ok = all(
                _EXPLICIT_REGION_REGISTRY.regions.get(region_id) is record
                and int(record.base) == base
                and int(record.length) == length
                for region_id, (record, base, length) in snapshots.items()
            )
            ok = ok and identity_ok
            for record, _base, _length in snapshots.values():
                record.state = RegionState.CPU if ok else RegionState.UNKNOWN

            return {
                "ok": ok,
                "reason": (
                    "ok"
                    if ok
                    else (
                        "remote_read_registry_changed"
                        if not identity_ok
                        else "remote_read_batch_audit_failed"
                    )
                ),
                "region_count": len(requested),
                "statuses": {region_id: ok for region_id in requested},
                "failed_region_ids": [] if ok else requested,
                "batch": True,
                "batch_result": batch_result,
            }

    result = release_uvm_regions(
        requested,
        device=int(device),
        remote_read=True,
        include_audit=True,
    )
    audits = result.get("audits")
    audit_ok = isinstance(audits, dict) and set(audits) == set(requested)
    if audit_ok:
        for region_id in requested:
            audit = audits.get(region_id)
            registered_length = (
                int(audit.get("registered_length", 0))
                if isinstance(audit, dict)
                else 0
            )
            if (
                not isinstance(audit, dict)
                or audit.get("ok") is not True
                or audit.get("remote_read") is not True
                or registered_length <= 0
                or int(audit.get("remote_map_failure_blocks", 0)) != 0
                or int(audit.get("cpu_resident_bytes", 0)) < registered_length
                or int(audit.get("remote_mapped_bytes", 0)) < registered_length
            ):
                audit_ok = False
                break
    ok = bool(result.get("ok")) and audit_ok
    return {
        **result,
        "ok": ok,
        "reason": "ok" if ok else "remote_read_audit_failed",
        "region_count": len(requested),
        "batch": False,
    }


def restore_uvm_regions(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    executor: str = "auto",
) -> dict[str, Any]:
    """Restore only the explicitly named regions using their registered ranges."""
    requested = [str(region_id) for region_id in region_ids]
    statuses: dict[str, bool] = {}
    pending: list[KernelRegion] = []
    with _EXPLICIT_REGION_LOCK:
        for region_id in requested:
            region = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            if region is None:
                statuses[region_id] = False
            elif region.state == RegionState.GPU:
                statuses[region_id] = True
            else:
                # UNKNOWN must be restored conservatively instead of being
                # treated as an implicit failure or an already-resident range.
                region.state = RegionState.PREFETCHING_GPU
                pending.append(region)

    if pending:
        ranges = [(region.base, region.length) for region in pending]
        try:
            successes, _, _ = batch_prefetch_ranges_to_device_with_executor(
                ranges,
                device,
                executor=executor,
            )
        except Exception:
            logger.warning("Failed to restore explicit UVM regions", exc_info=True)
            successes = 0
        complete = int(successes) == len(pending)
        with _EXPLICIT_REGION_LOCK:
            if complete:
                complete = _set_uvm_region_state_locked(
                    pending,
                    RegionState.GPU,
                )
            else:
                _set_uvm_region_state_locked(pending, RegionState.UNKNOWN)
            for region in pending:
                statuses[region.region_id] = complete

    return {
        "ok": all(statuses.get(region_id, False) for region_id in requested),
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses.get(region_id, False)
        ],
    }


def _slot_direct_trace_cache_fields(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, dict[str, int]]:
    return {
        "cache_before": {
            "bytes": int(before["allocated_bytes"]),
            "slots": int(before["allocated_slots"]),
        },
        "cache_after": {
            "bytes": int(after["allocated_bytes"]),
            "slots": int(after["allocated_slots"]),
        },
        "limits": {
            "bytes": int(after["cache_bytes_limit"]),
            "max_slots": int(after["cache_slots_limit"]),
        },
    }


def release_uvm_regions_from_slots(
    region_ids: list[str] | tuple[str, ...],
    *,
    layer_id: int,
    expert_id: int,
    device: int = 0,
    trigger: str = "explicit",
) -> dict[str, Any]:
    """Release slot PTEs/backing and reconcile policy state entry by entry.

    The caller must establish a CUDA completion boundary before releasing any
    slot which a user kernel may still read. The vLLM fused-MoE call site does
    this on the single-GPU execution stream before it can invoke capacity
    eviction.
    """
    if trigger not in ("capacity", "explicit"):
        raise ValueError(f"Unsupported slot release trigger: {trigger!r}")
    with _SLOT_DIRECT_OPERATION_LOCK:
        return _release_uvm_regions_from_slots_impl(
            region_ids,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
            trigger=trigger,
        )


def _release_uvm_regions_from_slots_impl(
    region_ids: list[str] | tuple[str, ...],
    *,
    layer_id: int,
    expert_id: int,
    device: int,
    trigger: str,
) -> dict[str, Any]:
    global _SLOT_DIRECT_POISON_REASON, _SLOT_DIRECT_FIXED_BANK_LANES

    requested = list(dict.fromkeys(str(region_id) for region_id in region_ids))
    started_ns = time.perf_counter_ns()
    keys = [
        _slot_direct_binding_key(layer_id, expert_id, region_id)
        for region_id in requested
    ]
    with _SLOT_DIRECT_LOCK:
        before = _slot_direct_cache_snapshot_locked()
        slot_ids = {
            region_id: _SLOT_DIRECT_BINDINGS[key]
            for region_id, key in zip(requested, keys)
            if key in _SLOT_DIRECT_BINDINGS
        }
        active = [
            (
                region_id,
                key,
                _slot_direct_driver_identity_locked(key)[0],
                _slot_direct_driver_owner_token_locked(layer_id, expert_id),
                _SLOT_DIRECT_BINDINGS[key],
            )
            for region_id, key in zip(requested, keys)
            if key in _SLOT_DIRECT_PREALLOCATED
            and key in _SLOT_DIRECT_BINDINGS
        ]
        owner_thread = _SLOT_DIRECT_OWNER_THREAD

    if (
        active
        and owner_thread is not None
        and owner_thread != threading.get_ident()
    ):
        result = {
            "ok": False,
            "reason": "slot_direct_single_execution_thread_only",
            "trigger": trigger,
            "layer_id": int(layer_id),
            "expert_id": int(expert_id),
            "region_ids": requested,
            "slot_ids": slot_ids,
            "statuses": {region_id: False for region_id in requested},
            "failed_region_ids": requested,
            "result": None,
            "logical_placement": "slot_gpu",
            "ordinary_uvm_placement": "cpu",
            "poisoned": False,
            "poison_reason": None,
            **_slot_direct_trace_cache_fields(before, before),
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        _write_slot_direct_trace({"operation": "release", **result})
        return result

    release_result: dict[str, object] | None = None
    released_keys: set[tuple[int, int, str]] = set()
    if active:
        try:
            release_result = driver_slot_release_ranges(
                [
                    (int(driver_layer), int(driver_owner), int(slot_id))
                    for _region_id, _key, driver_layer, driver_owner, slot_id in active
                ],
                device=int(device),
            )
        except Exception as exc:
            release_result = {
                "ok": False,
                "reason": "slot_release_exception",
                "error": repr(exc),
                "entry_statuses": [None] * len(active),
            }
        raw_statuses = release_result.get("entry_statuses")
        if isinstance(raw_statuses, list) and len(raw_statuses) == len(active):
            for (
                _region_id,
                key,
                _driver_layer,
                _driver_owner,
                _slot_id,
            ), status in zip(
                active, raw_statuses
            ):
                if type(status) is int and status == 0:
                    released_keys.add(key)
        elif bool(release_result.get("ok")):
            released_keys.update(
                key
                for _region_id, key, _driver_layer, _driver_owner, _slot_id in active
            )

    active_keys = {
        key
        for _region_id, key, _driver_layer, _driver_owner, _slot_id in active
    }
    statuses: dict[str, bool] = {}
    with _SLOT_DIRECT_LOCK:
        if released_keys:
            _SLOT_DIRECT_FIXED_BANK_LANES = None
        for region_id, key in zip(requested, keys):
            success = key not in active_keys or key in released_keys
            statuses[region_id] = success
            if success:
                _SLOT_DIRECT_READY.discard(key)
                _SLOT_DIRECT_PREALLOCATED.discard(key)
                _SLOT_DIRECT_ALLOCATION_BYTES.pop(key, None)
                if trigger == "explicit":
                    _slot_direct_forget_key_locked(key)

        group = (int(layer_id), int(expert_id))
        if not any(
            (key[0], key[1]) == group for key in _SLOT_DIRECT_PREALLOCATED
        ):
            _SLOT_DIRECT_GROUP_LAST_USED.pop(group, None)
        after = _slot_direct_cache_snapshot_locked()

    ok = all(statuses.values()) and (
        release_result is None or bool(release_result.get("ok"))
    )
    reason = (
        "already_slot_released"
        if not active
        else "ok" if ok else "slot_release_failed"
    )
    if active and not ok:
        with _SLOT_DIRECT_LOCK:
            _SLOT_DIRECT_POISON_REASON = (
                "mode19 release did not reconcile every tracked slot: "
                f"layer={layer_id}, expert={expert_id}, trigger={trigger}"
            )
    result: dict[str, Any] = {
        "ok": ok,
        "reason": reason,
        "trigger": trigger,
        "layer_id": int(layer_id),
        "expert_id": int(expert_id),
        "region_ids": requested,
        "slot_ids": slot_ids,
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses[region_id]
        ],
        "result": release_result,
        "logical_placement": "cpu" if ok else "partial",
        "ordinary_uvm_placement": "cpu",
        "poisoned": _SLOT_DIRECT_POISON_REASON is not None,
        "poison_reason": _SLOT_DIRECT_POISON_REASON,
        **_slot_direct_trace_cache_fields(before, after),
        "wall_ns": time.perf_counter_ns() - started_ns,
    }
    _write_slot_direct_trace({"operation": "release", **result})
    return result


def close_uvm_slot_direct_policy_session(*, device: int = 0) -> dict[str, Any]:
    """Close the driver session and atomically forget all logical slot state."""
    started_ns = time.perf_counter_ns()
    with _SLOT_DIRECT_OPERATION_LOCK:
        with _SLOT_DIRECT_LOCK:
            before = _slot_direct_cache_snapshot_locked()
            active_keys = sorted(_SLOT_DIRECT_PREALLOCATED)
            region_ids = [key[2] for key in active_keys]
            slot_ids = {
                key[2]: _SLOT_DIRECT_BINDINGS[key]
                for key in active_keys
                if key in _SLOT_DIRECT_BINDINGS
            }
        close_result = close_uvm_slot_direct_session(device=int(device))
        with _SLOT_DIRECT_LOCK:
            after = _slot_direct_cache_snapshot_locked()

    ok = bool(close_result.get("ok")) and after["allocated_slots"] == 0
    result: dict[str, Any] = {
        "ok": ok,
        "reason": "closed" if ok else "slot_session_close_failed",
        "trigger": "explicit",
        "scope": "session",
        "layer_id": -1,
        "expert_id": -1,
        "region_ids": region_ids,
        "slot_ids": slot_ids,
        "statuses": {region_id: ok for region_id in region_ids},
        "failed_region_ids": [] if ok else region_ids,
        "result": close_result,
        "logical_placement": "cpu" if ok else "partial",
        "ordinary_uvm_placement": "cpu",
        **_slot_direct_trace_cache_fields(before, after),
        "wall_ns": time.perf_counter_ns() - started_ns,
    }
    _write_slot_direct_trace({"operation": "release", **result})
    return result


def _slot_direct_capacity_victims_locked(
    keys: list[tuple[int, int, str]],
    entries_by_key: dict[tuple[int, int, str], tuple[int, ...]],
) -> tuple[list[tuple[int, int, list[str]]], str]:
    requested_group = (int(keys[0][0]), int(keys[0][1]))
    return _slot_direct_capacity_victims_for_groups_locked(
        keys,
        entries_by_key,
        {requested_group},
        scope_name="one expert",
    )


def _slot_direct_capacity_victims_for_groups_locked(
    keys: list[tuple[int, int, str]],
    entries_by_key: dict[tuple[int, int, str], tuple[int, ...]],
    requested_groups: set[tuple[int, int]],
    *,
    scope_name: str,
) -> tuple[list[tuple[int, int, list[str]]], str]:
    """Select victims once for an atomic upcoming working set.

    ``keys`` contains every requested key, including keys which are already
    resident. Keeping the complete group set visible here prevents a layer
    batch from selecting one of its own experts as a capacity victim.
    """
    if not keys or not requested_groups:
        return [], ""

    cache_bytes, max_slots = _slot_direct_cache_limits_locked()
    requested_group_keys = {
        key
        for key in _SLOT_DIRECT_PREALLOCATED
        if (key[0], key[1]) in requested_groups
    } | set(keys)
    requested_group_bytes = sum(
        int(
            _SLOT_DIRECT_ALLOCATION_BYTES[key]
            if key in _SLOT_DIRECT_ALLOCATION_BYTES
            else entries_by_key[key][1]
        )
        for key in requested_group_keys
    )
    if requested_group_bytes > cache_bytes:
        return [], (
            f"{scope_name} requires more bytes than the bounded slot cache: "
            f"requested={requested_group_bytes}, limit={cache_bytes}"
        )
    if len(requested_group_keys) > max_slots:
        return [], (
            f"{scope_name} requires more slots than the bounded slot cache: "
            f"requested={len(requested_group_keys)}, limit={max_slots}"
        )

    additional_keys = [
        key for key in keys if key not in _SLOT_DIRECT_PREALLOCATED
    ]
    projected_bytes = sum(_SLOT_DIRECT_ALLOCATION_BYTES.values()) + sum(
        int(entries_by_key[key][1]) for key in additional_keys
    )
    projected_slots = len(_SLOT_DIRECT_PREALLOCATED) + len(additional_keys)
    if projected_bytes <= cache_bytes and projected_slots <= max_slots:
        return [], ""

    groups: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
    for key in _SLOT_DIRECT_PREALLOCATED:
        group = (int(key[0]), int(key[1]))
        if (
            group not in requested_groups
            and group not in _SLOT_DIRECT_PROTECTED_GROUPS
        ):
            groups.setdefault(group, []).append(key)
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            _SLOT_DIRECT_GROUP_LAST_USED.get(group, 0),
            group,
        ),
    )

    victims: list[tuple[int, int, list[str]]] = []
    for group in ordered_groups:
        group_keys = groups[group]
        victims.append(
            (group[0], group[1], sorted(key[2] for key in group_keys))
        )
        projected_bytes -= sum(
            _SLOT_DIRECT_ALLOCATION_BYTES.get(key, 0) for key in group_keys
        )
        projected_slots -= len(group_keys)
        if projected_bytes <= cache_bytes and projected_slots <= max_slots:
            return victims, ""
    return [], (
        "bounded slot cache has no safe capacity victim: "
        f"projected_bytes={projected_bytes}, byte_limit={cache_bytes}, "
        f"projected_slots={projected_slots}, slot_limit={max_slots}"
    )


def _slot_direct_rebind_capacity_victims_locked(
    keys: list[tuple[int, int, str]],
    entries_by_key: dict[tuple[int, int, str], tuple[int, ...]],
    victims: list[tuple[int, int, list[str]]],
) -> tuple[
    dict[
        tuple[int, int, str],
        tuple[tuple[int, int, str], int, int, int],
    ],
    list[dict[str, Any]],
    str,
]:
    """Transfer an exactly compatible physical bank lease to new owners.

    The kernel mode-8 rebind is keyed by ``(driver_layer, slot_id)``. This
    function changes only userspace ownership metadata; the following mode-8
    call carries ``REUSE_BACKING_REQUIRED`` and performs the old-PTE teardown
    plus owner replacement atomically under the UVM VA-space/pool locks.
    """
    global _SLOT_DIRECT_FIXED_BANK_LANES
    global _SLOT_DIRECT_FIXED_BANK_PLAN_HITS
    global _SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS
    global _SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS

    if not _slot_direct_reuse_bank_enabled() or not victims:
        return {}, [], "disabled_or_no_victim"

    new_keys = [key for key in keys if key not in _SLOT_DIRECT_PREALLOCATED]
    wanted_by_group = {
        (int(victim_layer), int(victim_expert)): set(victim_region_ids)
        for victim_layer, victim_expert, victim_region_ids in victims
    }
    old_keys_by_group: dict[
        tuple[int, int], list[tuple[int, int, str]]
    ] = {group: [] for group in wanted_by_group}
    # Build the victim index once. The former implementation scanned the
    # complete bank once per expert (128 * 1,152 membership checks).
    for key in _SLOT_DIRECT_PREALLOCATED:
        group = (int(key[0]), int(key[1]))
        wanted = wanted_by_group.get(group)
        if wanted is not None and key[2] in wanted:
            old_keys_by_group[group].append(key)

    victim_keys: list[tuple[int, int, str]] = []
    for victim_layer, victim_expert, victim_region_ids in victims:
        group = (int(victim_layer), int(victim_expert))
        wanted = set(victim_region_ids)
        matched = old_keys_by_group[group]
        if {key[2] for key in matched} != wanted:
            return {}, [], "victim_state_mismatch"
        victim_keys.extend(matched)

    if len(victim_keys) != len(new_keys):
        return {}, [], "slot_count_mismatch"

    victim_key_set = set(victim_keys)
    complete_bank = (
        len(victim_key_set) == len(_SLOT_DIRECT_PREALLOCATED)
        and victim_key_set == _SLOT_DIRECT_PREALLOCATED
    )
    old_by_size: list[tuple[int, int, str]] | None = None
    lane_plan_hit = False
    lane_plan_build_started_ns = 0
    precomputed_lanes = _slot_direct_precomputed_lanes_enabled()
    if (
        complete_bank
        and precomputed_lanes
        and _SLOT_DIRECT_FIXED_BANK_LANES is not None
    ):
        cached_old_by_size: list[tuple[int, int, str]] = []
        cached_owners: set[tuple[int, int, str]] = set()
        for driver_layer, slot_id, allocation_bytes in _SLOT_DIRECT_FIXED_BANK_LANES:
            old_key = _SLOT_DIRECT_SLOT_OWNERS.get((driver_layer, slot_id))
            if (
                old_key is None
                or old_key not in victim_key_set
                or _SLOT_DIRECT_ALLOCATION_BYTES.get(old_key) != allocation_bytes
            ):
                cached_old_by_size = []
                break
            cached_old_by_size.append(old_key)
            cached_owners.add(old_key)
        if (
            len(cached_old_by_size) == len(victim_keys)
            and cached_owners == victim_key_set
        ):
            old_by_size = cached_old_by_size
            lane_plan_hit = True
        else:
            # A release or non-standard transition changed the physical bank.
            # Relearn only from fully validated live policy state below.
            _SLOT_DIRECT_FIXED_BANK_LANES = None

    if old_by_size is None:
        if complete_bank and precomputed_lanes:
            lane_plan_build_started_ns = time.perf_counter_ns()
        old_by_size = sorted(
            victim_keys,
            key=lambda key: (_SLOT_DIRECT_ALLOCATION_BYTES.get(key, -1), key),
        )
    new_by_size = sorted(
        new_keys,
        key=lambda key: (int(entries_by_key[key][1]), key),
    )
    if any(
        _SLOT_DIRECT_ALLOCATION_BYTES.get(old_key) != int(entries_by_key[new_key][1])
        for old_key, new_key in zip(old_by_size, new_by_size)
    ):
        # Keep accounting exact: a larger retained backing must not be reported
        # as only the smaller logical request. Non-identical shapes use the
        # existing audited release/allocation path.
        return {}, [], "slot_size_mismatch"

    transfers: list[
        tuple[
            tuple[int, int, str],
            tuple[int, int, str],
            int,
            int,
            int,
        ]
    ] = []
    learned_lanes: list[tuple[int, int, int]] = []
    for old_key, new_key in zip(old_by_size, new_by_size):
        driver_layer, slot_id = _slot_direct_driver_identity_locked(old_key)
        if driver_layer != _SLOT_DIRECT_REUSE_BANK_LAYER:
            return {}, [], "driver_namespace_mismatch"
        if _SLOT_DIRECT_SLOT_OWNERS.get((driver_layer, slot_id)) != old_key:
            return {}, [], "physical_owner_mismatch"
        transfers.append(
            (
                old_key,
                new_key,
                driver_layer,
                slot_id,
                _SLOT_DIRECT_ALLOCATION_BYTES[old_key],
            )
        )
        learned_lanes.append(
            (
                int(driver_layer),
                int(slot_id),
                int(_SLOT_DIRECT_ALLOCATION_BYTES[old_key]),
            )
        )

    if (
        complete_bank
        and precomputed_lanes
        and _SLOT_DIRECT_FIXED_BANK_LANES is None
    ):
        # The identities and sizes remain valid across owner changes. Owner
        # keys are deliberately not cached and are re-read from the
        # authoritative lease table for every transition.
        _SLOT_DIRECT_FIXED_BANK_LANES = tuple(learned_lanes)
        _SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS += 1
        _SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS += (
            time.perf_counter_ns() - lane_plan_build_started_ns
        )
    elif lane_plan_hit:
        _SLOT_DIRECT_FIXED_BANK_PLAN_HITS += 1

    plan_by_new: dict[
        tuple[int, int, str],
        tuple[tuple[int, int, str], int, int, int],
    ] = {}
    transfer_by_old: dict[tuple[int, int, str], dict[str, Any]] = {}
    for (
        old_key,
        new_key,
        driver_layer,
        slot_id,
        allocation_bytes,
    ) in transfers:
        plan_by_new[new_key] = (
            old_key,
            driver_layer,
            slot_id,
            allocation_bytes,
        )
        transfer_by_old[old_key] = {
            "old_region_id": old_key[2],
            "new_layer_id": int(new_key[0]),
            "new_expert_id": int(new_key[1]),
            "new_region_id": new_key[2],
            "driver_layer_id": driver_layer,
            "slot_id": slot_id,
            "bytes": allocation_bytes,
        }

    ordered_old_by_group: dict[
        tuple[int, int], list[tuple[int, int, str]]
    ] = {}
    for key in old_by_size:
        ordered_old_by_group.setdefault((int(key[0]), int(key[1])), []).append(key)

    eviction_records: list[dict[str, Any]] = []
    for victim_layer, victim_expert, victim_region_ids in victims:
        group_keys = ordered_old_by_group.get(
            (int(victim_layer), int(victim_expert)), []
        )
        eviction_records.append(
            {
                "layer_id": victim_layer,
                "expert_id": victim_expert,
                "region_ids": list(victim_region_ids),
                "slot_ids": {
                    key[2]: int(transfer_by_old[key]["slot_id"])
                    for key in group_keys
                },
                "release": None,
                "mechanism": "bank_rebind",
                "rebind_targets": [transfer_by_old[key] for key in group_keys],
            }
        )
    return plan_by_new, eviction_records, "ok"


def _slot_direct_commit_rebinds_locked(
    plan_by_new: dict[
        tuple[int, int, str],
        tuple[tuple[int, int, str], int, int, int],
    ],
    successful_keys: set[tuple[int, int, str]],
) -> None:
    """Commit only mode-8 entries whose kernel status proved success."""
    successful_transfers = [
        (new_key, *plan_by_new[new_key])
        for new_key in successful_keys
    ]
    # Validate the complete trusted status subset before mutating any
    # userspace lease. This preserves the partial-mode8 fail-closed contract
    # while avoiding a half-applied userspace commit on an internal mismatch.
    for (
        _new_key,
        old_key,
        driver_layer,
        slot_id,
        _allocation_bytes,
    ) in successful_transfers:
        if _SLOT_DIRECT_SLOT_OWNERS.get((driver_layer, slot_id)) != old_key:
            raise RuntimeError("slot bank owner changed during serialized rebind")

    old_groups: set[tuple[int, int]] = set()
    for (
        new_key,
        old_key,
        driver_layer,
        slot_id,
        _allocation_bytes,
    ) in successful_transfers:
        provisional_slot = _SLOT_DIRECT_BINDINGS.pop(new_key, None)
        provisional_layer = _SLOT_DIRECT_DRIVER_LAYERS.pop(
            new_key, _SLOT_DIRECT_REUSE_BANK_LAYER
        )
        if provisional_slot is not None:
            provisional_identity = (provisional_layer, provisional_slot)
            if _SLOT_DIRECT_SLOT_OWNERS.get(provisional_identity) == new_key:
                _SLOT_DIRECT_SLOT_OWNERS.pop(provisional_identity, None)

        _SLOT_DIRECT_READY.discard(old_key)
        _SLOT_DIRECT_PREALLOCATED.discard(old_key)
        _SLOT_DIRECT_ALLOCATION_BYTES.pop(old_key, None)
        _SLOT_DIRECT_BINDINGS.pop(old_key, None)
        _SLOT_DIRECT_DRIVER_LAYERS.pop(old_key, None)
        _SLOT_DIRECT_SLOT_OWNERS.pop((driver_layer, slot_id), None)

        _SLOT_DIRECT_BINDINGS[new_key] = slot_id
        _SLOT_DIRECT_DRIVER_LAYERS[new_key] = driver_layer
        _SLOT_DIRECT_SLOT_OWNERS[(driver_layer, slot_id)] = new_key

        old_groups.add((int(old_key[0]), int(old_key[1])))

    # The former per-lane any(PREALLOCATED) check was O(bank_slots^2) for a
    # complete layer transition. Compute all surviving groups in one pass.
    surviving_groups = {
        (int(key[0]), int(key[1])) for key in _SLOT_DIRECT_PREALLOCATED
    }
    for old_group in old_groups:
        if old_group not in surviving_groups:
            _SLOT_DIRECT_GROUP_LAST_USED.pop(old_group, None)


def restore_uvm_regions_to_slots(
    region_ids: list[str] | tuple[str, ...],
    *,
    layer_id: int,
    expert_id: int,
    device: int = 0,
) -> dict[str, Any]:
    with _SLOT_DIRECT_OPERATION_LOCK:
        return _restore_uvm_regions_to_slots_impl(
            region_ids,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
        )


def _restore_uvm_regions_to_slots_impl(
    region_ids: list[str] | tuple[str, ...],
    *,
    layer_id: int,
    expert_id: int,
    device: int,
) -> dict[str, Any]:
    """Restore immutable weight regions through preallocated VIDMEM slots.

    The source remains ordinary CPU-resident UVM memory. A successful call
    installs read-only slot PTEs at the original addresses and records logical
    readiness separately from :class:`RegionState`. Both mode 8 and mode 10
    carry ``NO_EVICT`` so the kernel must fail instead of silently invalidating
    another expert mapping that userspace still considers ready.
    """
    global _SLOT_DIRECT_ACTIVE_DEVICE, _SLOT_DIRECT_OWNER_THREAD
    global _SLOT_DIRECT_POISON_REASON

    requested = list(dict.fromkeys(str(region_id) for region_id in region_ids))
    started_ns = time.perf_counter_ns()
    statuses = {region_id: False for region_id in requested}
    with _SLOT_DIRECT_LOCK:
        initial_cache = _slot_direct_cache_snapshot_locked()
    if not requested:
        result = {
            "ok": True,
            "reason": "empty",
            "statuses": statuses,
            "failed_region_ids": [],
            "logical_placement": "slot_gpu",
            "ordinary_uvm_placement": "cpu",
            "requested_bytes": 0,
            "evictions": [],
            **_slot_direct_trace_cache_fields(initial_cache, initial_cache),
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        _write_slot_direct_trace(
            {
                "operation": "restore",
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
                **result,
            }
        )
        return result

    with _EXPLICIT_REGION_LOCK:
        records = {
            region_id: _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            for region_id in requested
        }
    missing = [region_id for region_id, region in records.items() if region is None]
    if missing:
        result = {
            "ok": False,
            "reason": "unregistered_region",
            "statuses": statuses,
            "failed_region_ids": missing,
            "logical_placement": "unknown",
            "ordinary_uvm_placement": "unchanged",
            "requested_bytes": 0,
            "evictions": [],
            **_slot_direct_trace_cache_fields(initial_cache, initial_cache),
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        _write_slot_direct_trace(
            {
                "operation": "restore",
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
                **result,
            }
        )
        return result

    keys = [
        _slot_direct_binding_key(layer_id, expert_id, region_id)
        for region_id in requested
    ]
    with _SLOT_DIRECT_LOCK:
        if _SLOT_DIRECT_POISON_REASON is not None:
            result = {
                "ok": False,
                "reason": "slot_direct_session_poisoned",
                "error": _SLOT_DIRECT_POISON_REASON,
                "statuses": statuses,
                "failed_region_ids": requested,
                "logical_placement": "unknown",
                "ordinary_uvm_placement": "cpu",
                "requested_bytes": 0,
                "evictions": [],
                **_slot_direct_trace_cache_fields(initial_cache, initial_cache),
                "wall_ns": time.perf_counter_ns() - started_ns,
            }
            _write_slot_direct_trace(
                {
                    "operation": "restore",
                    "layer_id": int(layer_id),
                    "expert_id": int(expert_id),
                    **result,
                }
            )
            return result
        current_thread = threading.get_ident()
        if (
            _SLOT_DIRECT_OWNER_THREAD is not None
            and _SLOT_DIRECT_OWNER_THREAD != current_thread
        ):
            result = {
                "ok": False,
                "reason": "slot_direct_single_execution_thread_only",
                "statuses": statuses,
                "failed_region_ids": requested,
                "logical_placement": "unknown",
                "ordinary_uvm_placement": "cpu",
                "requested_bytes": 0,
                "evictions": [],
                **_slot_direct_trace_cache_fields(initial_cache, initial_cache),
                "wall_ns": time.perf_counter_ns() - started_ns,
            }
            _write_slot_direct_trace(
                {
                    "operation": "restore",
                    "layer_id": int(layer_id),
                    "expert_id": int(expert_id),
                    **result,
                }
            )
            return result
        _SLOT_DIRECT_OWNER_THREAD = current_thread
        if (
            _SLOT_DIRECT_ACTIVE_DEVICE is not None
            and _SLOT_DIRECT_ACTIVE_DEVICE != int(device)
        ):
            result = {
                "ok": False,
                "reason": "slot_direct_single_gpu_only",
                "statuses": statuses,
                "failed_region_ids": requested,
                "logical_placement": "unknown",
                "ordinary_uvm_placement": "cpu",
                "requested_bytes": 0,
                "evictions": [],
                **_slot_direct_trace_cache_fields(initial_cache, initial_cache),
                "wall_ns": time.perf_counter_ns() - started_ns,
            }
            _write_slot_direct_trace(
                {
                    "operation": "restore",
                    "layer_id": int(layer_id),
                    "expert_id": int(expert_id),
                    **result,
                }
            )
            return result
        _SLOT_DIRECT_ACTIVE_DEVICE = int(device)
        slot_ids = {
            key: _slot_direct_slot_id_locked(key)
            for key in keys
        }
        already_ready = all(key in _SLOT_DIRECT_READY for key in keys)
        needs_preallocate = [
            key for key in keys if key not in _SLOT_DIRECT_PREALLOCATED
        ]

    requested_bytes = sum(int(records[region_id].length) for region_id in requested)  # type: ignore[union-attr]
    if already_ready:
        with _SLOT_DIRECT_LOCK:
            _slot_direct_touch_group_locked(layer_id, expert_id)
            final_cache = _slot_direct_cache_snapshot_locked()
        statuses = {region_id: True for region_id in requested}
        result = {
            "ok": True,
            "reason": "already_slot_ready",
            "statuses": statuses,
            "failed_region_ids": [],
            "logical_placement": "slot_gpu",
            "ordinary_uvm_placement": "cpu",
            "requested_bytes": requested_bytes,
            "preallocate": None,
            "restore": None,
            "evictions": [],
            "slot_ids": {
                region_id: slot_ids[key]
                for region_id, key in zip(requested, keys)
            },
            **_slot_direct_trace_cache_fields(initial_cache, final_cache),
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        _write_slot_direct_trace(
            {
                "operation": "restore",
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
                **result,
            }
        )
        return result

    entries_by_key: dict[tuple[int, int, str], tuple[int, ...]] = {}
    for region_id, key in zip(requested, keys):
        region = records[region_id]
        assert region is not None
        with _SLOT_DIRECT_LOCK:
            driver_layer, driver_slot = _slot_direct_driver_identity_locked(key)
            driver_owner = _slot_direct_driver_owner_token_locked(
                layer_id, expert_id
            )
        entries_by_key[key] = (
            int(region.base),
            int(region.length),
            int(driver_layer),
            int(driver_owner),
            int(driver_slot),
            0,
            _SLOT_DIRECT_NO_EVICT_FLAG,
        )

    rebind_plan: dict[
        tuple[int, int, str],
        tuple[tuple[int, int, str], int, int, int],
    ] = {}
    rebound_keys: set[tuple[int, int, str]] = set()
    rebind_reason = "not_attempted"
    evictions: list[dict[str, Any]] = []
    with _SLOT_DIRECT_LOCK:
        victims, capacity_error = _slot_direct_capacity_victims_locked(
            keys, entries_by_key
        )
        if not capacity_error:
            rebind_plan, rebind_evictions, rebind_reason = (
                _slot_direct_rebind_capacity_victims_locked(
                    keys, entries_by_key, victims
                )
            )
            rebound_keys = set(rebind_plan)
            evictions.extend(rebind_evictions)
            slot_ids = {
                key: (
                    rebind_plan[key][2]
                    if key in rebind_plan
                    else _SLOT_DIRECT_BINDINGS[key]
                )
                for key in keys
            }

    if capacity_error:
        with _SLOT_DIRECT_LOCK:
            final_cache = _slot_direct_cache_snapshot_locked()
        result = {
            "ok": False,
            "reason": "slot_cache_capacity_failed",
            "error": capacity_error,
            "statuses": statuses,
            "failed_region_ids": requested,
            "logical_placement": "unknown",
            "ordinary_uvm_placement": "cpu",
            "requested_bytes": requested_bytes,
            "preallocate": None,
            "restore": None,
            "evictions": evictions,
            "slot_ids": {
                region_id: slot_ids[key]
                for region_id, key in zip(requested, keys)
            },
            **_slot_direct_trace_cache_fields(initial_cache, final_cache),
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        _write_slot_direct_trace(
            {
                "operation": "restore",
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
                **result,
            }
        )
        return result

    # A successful bank transfer deliberately skips mode 19. Mode 8 will clear
    # each victim's old PTE and rebind the retained backing to the new VA.
    release_victims = [] if rebound_keys else victims
    for victim_layer, victim_expert, victim_region_ids in release_victims:
        release = release_uvm_regions_from_slots(
            victim_region_ids,
            layer_id=victim_layer,
            expert_id=victim_expert,
            device=int(device),
            trigger="capacity",
        )
        eviction = {
            "layer_id": victim_layer,
            "expert_id": victim_expert,
            "region_ids": victim_region_ids,
            "slot_ids": dict(release.get("slot_ids", {})),
            "release": release,
        }
        evictions.append(eviction)
        if not bool(release.get("ok")):
            with _SLOT_DIRECT_LOCK:
                final_cache = _slot_direct_cache_snapshot_locked()
            result = {
                "ok": False,
                "reason": "slot_cache_eviction_failed",
                "statuses": statuses,
                "failed_region_ids": requested,
                "logical_placement": "unknown",
                "ordinary_uvm_placement": "cpu",
                "requested_bytes": requested_bytes,
                "preallocate": None,
                "restore": None,
                "evictions": evictions,
                "slot_ids": {
                    region_id: slot_ids[key]
                    for region_id, key in zip(requested, keys)
                },
                **_slot_direct_trace_cache_fields(initial_cache, final_cache),
                "wall_ns": time.perf_counter_ns() - started_ns,
            }
            _write_slot_direct_trace(
                {
                    "operation": "restore",
                    "layer_id": int(layer_id),
                    "expert_id": int(expert_id),
                    **result,
                }
            )
            return result

    # Capacity handling may have replaced provisional logical identities with
    # retained physical-bank identities. Rebuild both mode-8 and mode-10
    # entries from the committed lease table. The transient reuse requirement
    # is sent only to mode 8; mode 10 accepts lifetime flags only.
    entries_by_key = {}
    preallocate_entries_by_key: dict[
        tuple[int, int, str], tuple[int, ...]
    ] = {}
    with _SLOT_DIRECT_LOCK:
        driver_owner = _slot_direct_driver_owner_token_locked(layer_id, expert_id)
        for region_id, key in zip(requested, keys):
            region = records[region_id]
            assert region is not None
            if key in rebind_plan:
                (
                    _old_key,
                    driver_layer,
                    driver_slot,
                    _allocation_bytes,
                ) = (
                    rebind_plan[key]
                )
            else:
                driver_layer, driver_slot = _slot_direct_driver_identity_locked(
                    key
                )
            entry = (
                int(region.base),
                int(region.length),
                int(driver_layer),
                int(driver_owner),
                int(driver_slot),
                0,
                _SLOT_DIRECT_NO_EVICT_FLAG,
            )
            entries_by_key[key] = entry
            preallocate_entries_by_key[key] = (
                *entry[:-1],
                entry[-1]
                | (
                    UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
                    if key in rebound_keys
                    else 0
                ),
            )

    preallocate_result: dict[str, object] | None = None
    if needs_preallocate:
        preallocate_raised = False
        try:
            preallocate_result = driver_slot_preallocate_ranges(
                [preallocate_entries_by_key[key] for key in needs_preallocate],
                device=int(device),
            )
        except Exception as exc:
            preallocate_raised = True
            preallocate_result = {
                "ok": False,
                "reason": "slot_preallocate_exception",
                "error": repr(exc),
                "entry_statuses": [None] * len(needs_preallocate),
            }
        raw_statuses = preallocate_result.get("entry_statuses")
        if isinstance(raw_statuses, list) and len(raw_statuses) == len(
            needs_preallocate
        ):
            successful_preallocations = [
                key
                for key, status in zip(needs_preallocate, raw_statuses)
                if type(status) is int and status == 0
            ]
        elif bool(preallocate_result.get("ok")):
            successful_preallocations = list(needs_preallocate)
        else:
            successful_preallocations = []
        with _SLOT_DIRECT_LOCK:
            successful_rebinds = set(successful_preallocations) & rebound_keys
            _slot_direct_commit_rebinds_locked(
                rebind_plan, successful_rebinds
            )
            _SLOT_DIRECT_PREALLOCATED.update(successful_preallocations)
            _SLOT_DIRECT_ALLOCATION_BYTES.update(
                {
                    key: int(entries_by_key[key][1])
                    for key in successful_preallocations
                }
            )
        if not bool(preallocate_result.get("ok")):
            rollback: dict[str, Any] | None = None
            if rebound_keys:
                # Per-entry commits above mirror the kernel's exact statuses.
                # Do not run mode 19: it would destroy successfully retained
                # bank backing. Poison the serialized session and let final
                # close release the low-level owner table exactly once.
                with _SLOT_DIRECT_LOCK:
                    _SLOT_DIRECT_POISON_REASON = (
                        "mode8 bank rebind did not complete every slot: "
                        f"layer={layer_id}, expert={expert_id}"
                    )
            elif successful_preallocations:
                rollback = release_uvm_regions_from_slots(
                    requested,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    device=int(device),
                    trigger="explicit",
                )
            elif preallocate_raised:
                with _SLOT_DIRECT_LOCK:
                    _SLOT_DIRECT_POISON_REASON = (
                        "mode8 rebind failed with indeterminate driver ownership: "
                        f"layer={layer_id}, expert={expert_id}"
                    )
            with _SLOT_DIRECT_LOCK:
                final_cache = _slot_direct_cache_snapshot_locked()
            result = {
                "ok": False,
                "reason": "slot_preallocate_failed",
                "statuses": statuses,
                "failed_region_ids": requested,
                "logical_placement": "unknown",
                "ordinary_uvm_placement": "cpu",
                "requested_bytes": requested_bytes,
                "preallocate": preallocate_result,
                "restore": None,
                "rollback": rollback,
                "evictions": evictions,
                "slot_ids": {
                    region_id: slot_ids[key]
                    for region_id, key in zip(requested, keys)
                },
                **_slot_direct_trace_cache_fields(initial_cache, final_cache),
                "wall_ns": time.perf_counter_ns() - started_ns,
            }
            _write_slot_direct_trace(
                {
                    "operation": "restore",
                    "layer_id": int(layer_id),
                    "expert_id": int(expert_id),
                    **result,
                }
            )
            return result

    try:
        restore_result = driver_slot_restore_ranges(
            [entries_by_key[key] for key in keys],
            device=int(device),
        )
    except Exception as exc:
        restore_result = {
            "ok": False,
            "reason": "slot_restore_exception",
            "error": repr(exc),
            "entry_statuses": [None] * len(keys),
        }
    ok = bool(restore_result.get("ok"))
    rollback = None
    if ok:
        with _SLOT_DIRECT_LOCK:
            _SLOT_DIRECT_READY.update(keys)
            _slot_direct_touch_group_locked(layer_id, expert_id)
        statuses = {region_id: True for region_id in requested}
    elif rebound_keys:
        # The slot-direct helper has already rolled back recorded PTEs and
        # retained the bank backing. Releasing here would undo the optimization
        # and can race the driver's per-entry owner state after a partial
        # failure. Fail closed and release through session teardown only.
        with _SLOT_DIRECT_LOCK:
            _SLOT_DIRECT_POISON_REASON = (
                "mode10 failed after fixed-bank rebind: "
                f"layer={layer_id}, expert={expert_id}"
            )
    else:
        rollback = release_uvm_regions_from_slots(
            requested,
            layer_id=layer_id,
            expert_id=expert_id,
            device=int(device),
            trigger="explicit",
        )

    with _SLOT_DIRECT_LOCK:
        final_cache = _slot_direct_cache_snapshot_locked()

    result = {
        "ok": ok,
        "reason": "ok" if ok else "slot_restore_failed",
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses[region_id]
        ],
        "logical_placement": "slot_gpu" if ok else "unknown",
        "ordinary_uvm_placement": "cpu",
        "requested_bytes": requested_bytes,
        "preallocate": preallocate_result,
        "restore": restore_result,
        "rollback": rollback,
        "evictions": evictions,
        "bank_reuse_enabled": _slot_direct_reuse_bank_enabled(),
        "bank_rebind_slots": len(rebound_keys),
        "bank_rebind_reason": rebind_reason,
        "slot_ids": {
            region_id: slot_ids[key]
            for region_id, key in zip(requested, keys)
        },
        **_slot_direct_trace_cache_fields(initial_cache, final_cache),
        "wall_ns": time.perf_counter_ns() - started_ns,
    }
    _write_slot_direct_trace(
        {
            "operation": "restore",
            "layer_id": int(layer_id),
            "expert_id": int(expert_id),
            **result,
        }
    )
    return result


def restore_uvm_expert_groups_to_slots(
    expert_regions: Sequence[tuple[int, Sequence[str]]],
    *,
    layer_id: int,
    device: int = 0,
) -> dict[str, Any]:
    """Restore one fused-MoE working set with one layer-sized transaction.

    The low-level ABI accepts a complete 128-expert layer. Planning capacity,
    physical-bank transfers, and driver ownership for the whole layer before
    submission prevents the old per-expert loop from repeatedly validating the
    same session. Initial allocation uses mode 8 plus mode 10; an explicitly
    enabled complete fixed-bank rebind uses one fail-closed mode-20 ioctl.
    """
    with _SLOT_DIRECT_OPERATION_LOCK:
        return _restore_uvm_expert_groups_to_slots_impl(
            expert_regions,
            layer_id=int(layer_id),
            device=int(device),
            preallocate_only=False,
        )


def preallocate_uvm_expert_groups_to_slots(
    expert_regions: Sequence[tuple[int, Sequence[str]]],
    *,
    layer_id: int,
    device: int = 0,
) -> dict[str, Any]:
    """Build the fixed physical bank without copying or publishing PTEs.

    This is the initialization-time half of the first layer restore.  It uses
    mode 8 to reserve the immutable bank lanes, but deliberately leaves every
    logical region not-ready so the first request still performs the audited
    mode-21 copy/map operation.
    """
    with _SLOT_DIRECT_OPERATION_LOCK:
        return _restore_uvm_expert_groups_to_slots_impl(
            expert_regions,
            layer_id=int(layer_id),
            device=int(device),
            preallocate_only=True,
        )


def preallocate_uvm_registered_plan_second_bank(
    expert_regions: Sequence[tuple[int, Sequence[str]]],
    *,
    layer_id: int,
    device: int = 0,
) -> dict[str, Any]:
    """Allocate a disjoint second backing bank for registered plans only.

    Bank 0 is created by :func:`preallocate_uvm_expert_groups_to_slots`. This
    helper deliberately uses a separate driver-layer namespace and fresh slot
    identities, so the kernel must allocate different ``slot_mem`` objects.
    No PTE is published here. Any failure after the mode-8 submission poisons
    the policy session; the caller must close/abort it instead of falling back.
    """
    global _SLOT_DIRECT_POISON_REASON
    global _SLOT_DIRECT_REGISTERED_BANK_LANES
    global _SLOT_DIRECT_REGISTERED_BANK_KEYS

    started_ns = time.perf_counter_ns()
    try:
        dual_bank = _slot_direct_dual_bank_event_enabled()
        registered_plan = _slot_direct_registered_plan_enabled()
    except RuntimeError as exc:
        return {
            "ok": False,
            "reason": "dual_bank_config_invalid",
            "error": str(exc),
        }
    if not dual_bank:
        return {"ok": False, "reason": "dual_bank_event_disabled"}
    if not registered_plan:
        return {"ok": False, "reason": "dual_bank_requires_registered_plan"}

    normalized_groups: list[tuple[int, list[str]]] = []
    seen_experts: set[int] = set()
    seen_regions: set[str] = set()
    try:
        raw_groups = list(expert_regions)
    except TypeError:
        raw_groups = []
    for raw_group in raw_groups:
        try:
            expert_id, raw_region_ids = raw_group
            expert_id = int(expert_id)
            region_ids = [str(value) for value in raw_region_ids]
        except (TypeError, ValueError):
            return {"ok": False, "reason": "invalid_second_bank_groups"}
        if (
            expert_id in seen_experts
            or not region_ids
            or len(region_ids) != len(set(region_ids))
            or seen_regions.intersection(region_ids)
        ):
            return {"ok": False, "reason": "invalid_second_bank_groups"}
        seen_experts.add(expert_id)
        seen_regions.update(region_ids)
        normalized_groups.append((expert_id, region_ids))
    normalized_groups.sort(key=lambda item: item[0])
    if not normalized_groups:
        return {"ok": False, "reason": "empty_second_bank"}

    keys = [
        _slot_direct_binding_key(int(layer_id), expert_id, region_id)
        for expert_id, region_ids in normalized_groups
        for region_id in region_ids
    ]
    with _EXPLICIT_REGION_LOCK:
        records = {
            key[2]: _EXPLICIT_REGION_REGISTRY.regions.get(key[2]) for key in keys
        }
    if any(record is None for record in records.values()):
        return {"ok": False, "reason": "second_bank_region_missing"}
    record_lengths = {
        region_id: int(record.length)
        for region_id, record in records.items()
        if record is not None
    }

    with _SLOT_DIRECT_OPERATION_LOCK:
        staged: list[
            tuple[tuple[int, int, str], int, int, int, int, tuple[int, ...]]
        ] = []
        with _SLOT_DIRECT_LOCK:
            initial_cache = _slot_direct_cache_snapshot_locked()
            bank0_lanes = _SLOT_DIRECT_FIXED_BANK_LANES
            if _SLOT_DIRECT_POISON_REASON is not None:
                return {
                    "ok": False,
                    "reason": "slot_direct_session_poisoned",
                    "error": _SLOT_DIRECT_POISON_REASON,
                }
            if (
                _SLOT_DIRECT_ACTIVE_DEVICE != int(device)
                or _SLOT_DIRECT_OWNER_THREAD != threading.get_ident()
                or not bank0_lanes
                or _SLOT_DIRECT_READY
                or _SLOT_DIRECT_REGISTERED_BANK_LANES is not None
                or _SLOT_DIRECT_REGISTERED_BANK_KEYS is not None
                or _SLOT_DIRECT_REGISTERED_PLANS
            ):
                return {"ok": False, "reason": "second_bank_state_conflict"}
            if any(key in _SLOT_DIRECT_BINDINGS for key in keys):
                return {"ok": False, "reason": "second_bank_key_conflict"}

            bank0_keys: list[tuple[int, int, str]] = []
            for driver_layer, slot_id, _slot_bytes in bank0_lanes:
                owner = _SLOT_DIRECT_SLOT_OWNERS.get((driver_layer, slot_id))
                if owner is None:
                    return {"ok": False, "reason": "first_bank_owner_missing"}
                bank0_keys.append(owner)
            if len(set(bank0_keys)) != len(bank0_lanes):
                return {"ok": False, "reason": "first_bank_owner_duplicate"}

            ordered_keys = sorted(
                keys,
                key=lambda key: (record_lengths[key[2]], key),
            )
            bank0_sizes = [int(lane[2]) for lane in bank0_lanes]
            bank1_sizes = [record_lengths[key[2]] for key in ordered_keys]
            if bank1_sizes != bank0_sizes:
                return {"ok": False, "reason": "second_bank_geometry_mismatch"}

            cache_bytes, max_slots = _slot_direct_cache_limits_locked()
            total_bytes = sum(bank0_sizes) + sum(bank1_sizes)
            total_slots = len(bank0_sizes) + len(bank1_sizes)
            if total_bytes > cache_bytes or total_slots > max_slots:
                return {
                    "ok": False,
                    "reason": "dual_bank_capacity_exceeded",
                    "requested_bytes": total_bytes,
                    "requested_slots": total_slots,
                }

            reserved_identities = set(_SLOT_DIRECT_SLOT_OWNERS)
            for key in ordered_keys:
                record = records[key[2]]
                assert record is not None
                digest = hashlib.blake2s(
                    f"dual-bank:1:{key[0]}:{key[1]}:{key[2]}".encode("utf-8"),
                    digest_size=4,
                ).digest()
                slot_id = int.from_bytes(digest, "little") or 1
                first = slot_id
                while (_SLOT_DIRECT_SECOND_BANK_LAYER, slot_id) in reserved_identities:
                    slot_id = (slot_id + 1) & 0xFFFFFFFF
                    if slot_id == 0:
                        slot_id = 1
                    if slot_id == first:
                        return {"ok": False, "reason": "second_bank_slot_ids_exhausted"}
                identity = (_SLOT_DIRECT_SECOND_BANK_LAYER, slot_id)
                reserved_identities.add(identity)
                driver_owner = _slot_direct_driver_owner_token_locked(
                    int(layer_id), int(key[1])
                )
                entry = (
                    int(record.base),
                    int(record.length),
                    identity[0],
                    int(driver_owner),
                    identity[1],
                    0,
                    _SLOT_DIRECT_NO_EVICT_FLAG,
                )
                staged.append(
                    (
                        key,
                        identity[0],
                        identity[1],
                        int(record.length),
                        int(driver_owner),
                        entry,
                    )
                )

        entries = [item[5] for item in staged]
        try:
            preallocate_result = driver_slot_preallocate_ranges(
                entries, device=int(device)
            )
        except Exception as exc:
            preallocate_result = {
                "ok": False,
                "reason": "second_bank_preallocate_exception",
                "error": repr(exc),
            }
        statuses = preallocate_result.get("entry_statuses")
        complete = (
            bool(preallocate_result.get("ok"))
            and isinstance(statuses, list)
            and len(statuses) == len(staged)
            and all(type(value) is int and value == 0 for value in statuses)
        )
        if not complete:
            with _SLOT_DIRECT_LOCK:
                _SLOT_DIRECT_POISON_REASON = (
                    "second registered-plan backing bank did not allocate completely"
                )
            return {
                "ok": False,
                "reason": "second_bank_preallocate_failed",
                "preallocate": preallocate_result,
                "poisoned": True,
                "wall_ns": time.perf_counter_ns() - started_ns,
            }

        with _SLOT_DIRECT_LOCK:
            # The operation lock prevents policy mutation, but revalidate the
            # immutable first-bank recipe before publishing bank 1 in userspace.
            if (
                _SLOT_DIRECT_FIXED_BANK_LANES != bank0_lanes
                or _SLOT_DIRECT_REGISTERED_BANK_LANES is not None
                or _SLOT_DIRECT_REGISTERED_BANK_KEYS is not None
            ):
                _SLOT_DIRECT_POISON_REASON = (
                    "registered-plan bank state changed during second allocation"
                )
                return {
                    "ok": False,
                    "reason": "second_bank_postcondition_failed",
                    "preallocate": preallocate_result,
                    "poisoned": True,
                }
            bank1_lanes: list[tuple[int, int, int]] = []
            for key, driver_layer, slot_id, slot_bytes, _owner, _entry in staged:
                identity = (driver_layer, slot_id)
                _SLOT_DIRECT_BINDINGS[key] = slot_id
                _SLOT_DIRECT_DRIVER_LAYERS[key] = driver_layer
                _SLOT_DIRECT_SLOT_OWNERS[identity] = key
                _SLOT_DIRECT_PREALLOCATED.add(key)
                _SLOT_DIRECT_ALLOCATION_BYTES[key] = slot_bytes
                bank1_lanes.append((driver_layer, slot_id, slot_bytes))
            _SLOT_DIRECT_REGISTERED_BANK_LANES = (
                tuple(bank0_lanes),
                tuple(bank1_lanes),
            )
            _SLOT_DIRECT_REGISTERED_BANK_KEYS = (
                frozenset(bank0_keys),
                frozenset(keys),
            )
            final_cache = _slot_direct_cache_snapshot_locked()

        result = {
            "ok": True,
            "reason": "ok",
            "dual_bank_event": True,
            "bank_id": 1,
            "layer_id": int(layer_id),
            "lane_count": len(staged),
            "requested_bytes": sum(item[3] for item in staged),
            "preallocate": preallocate_result,
            **_slot_direct_trace_cache_fields(initial_cache, final_cache),
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        _write_slot_direct_trace({"operation": "preallocate_second_bank", **result})
        return result


@_trace_slot_direct_operation("register_layer_plans")
def register_uvm_slot_direct_layer_plans(
    layer_specs: Sequence[tuple[int, Sequence[tuple[int, Sequence[str]]]]],
    *,
    device: int = 0,
) -> dict[str, Any]:
    """Register immutable full-layer plans after the fixed bank exists."""
    global _SLOT_DIRECT_INITIAL_PLAN_ID, _SLOT_DIRECT_PLAN_GENERATION
    global _SLOT_DIRECT_PLAN_ACTIVE_LAYER
    global _SLOT_DIRECT_REGISTERED_RUNTIME

    started_ns = time.perf_counter_ns()
    if not _slot_direct_registered_plan_enabled():
        return {"ok": False, "reason": "registered_plan_disabled"}
    try:
        dual_bank_event = _slot_direct_dual_bank_event_enabled()
    except RuntimeError as exc:
        return {
            "ok": False,
            "reason": "dual_bank_config_invalid",
            "error": str(exc),
        }
    try:
        bank_level_commit = _slot_direct_bank_level_commit_enabled()
    except RuntimeError as exc:
        return {
            "ok": False,
            "reason": "bank_level_commit_config_invalid",
            "error": str(exc),
        }
    if bank_level_commit and not dual_bank_event:
        return {
            "ok": False,
            "reason": "bank_level_commit_requires_dual_bank",
        }

    with _SLOT_DIRECT_OPERATION_LOCK:
        normalized_specs = sorted(
            (
                int(layer_id),
                [
                    (int(expert_id), [str(region_id) for region_id in region_ids])
                    for expert_id, region_ids in groups
                ],
            )
            for layer_id, groups in layer_specs
        )
        if not normalized_specs:
            return {"ok": False, "reason": "empty_layer_plans"}
        layer_ids = [layer_id for layer_id, _groups in normalized_specs]
        if any(layer_id < 0 or layer_id > (1 << 32) - 1 for layer_id in layer_ids):
            return {"ok": False, "reason": "registered_plan_layer_id_invalid"}
        if len(layer_ids) != len(set(layer_ids)):
            return {"ok": False, "reason": "registered_plan_layer_id_duplicate"}
        if dual_bank_event:
            expected_layer_ids = list(
                range(min(layer_ids), min(layer_ids) + len(layer_ids))
            )
            if len(layer_ids) < 2 or len(layer_ids) % 2 != 0:
                return {
                    "ok": False,
                    "reason": "dual_bank_requires_even_layer_count",
                }
            if layer_ids != expected_layer_ids:
                return {
                    "ok": False,
                    "reason": "dual_bank_requires_contiguous_layers",
                    "layer_ids": layer_ids,
                }
        # The kernel registry contains at most 64 descriptors. One descriptor
        # is the bootstrap plan and every layer, including the initial layer,
        # has one transition descriptor.
        if len(normalized_specs) > 63:
            return {
                "ok": False,
                "reason": "registered_plan_limit_exceeded",
                "transition_plan_count": len(normalized_specs),
                "total_plan_count": len(normalized_specs) + 1,
                "max_plan_count": 64,
            }
        initial_plan_id = (min(layer_ids) << 2) | 1
        planned_ids = [initial_plan_id] + [
            (layer_id << 2) | 2 for layer_id in layer_ids
        ]
        if len(planned_ids) != len(set(planned_ids)):
            return {"ok": False, "reason": "registered_plan_id_duplicate"}

        with _SLOT_DIRECT_LOCK:
            if dual_bank_event:
                bank_lanes = _SLOT_DIRECT_REGISTERED_BANK_LANES
                bank_keys = _SLOT_DIRECT_REGISTERED_BANK_KEYS
                if (
                    bank_lanes is None
                    or bank_keys is None
                    or len(bank_lanes) != 2
                    or len(bank_keys) != 2
                    or not bank_lanes[0]
                    or len(bank_lanes[0]) != len(bank_lanes[1])
                    or set(bank_lanes[0]).intersection(bank_lanes[1])
                ):
                    return {"ok": False, "reason": "dual_bank_not_reserved"}
                lanes = bank_lanes[0]
                initial_keys = set(bank_keys[0])
                if set(bank_keys[0]).intersection(bank_keys[1]):
                    return {"ok": False, "reason": "dual_bank_key_overlap"}
                if set(_SLOT_DIRECT_PREALLOCATED) != set().union(*bank_keys):
                    return {"ok": False, "reason": "dual_bank_owner_mismatch"}
            else:
                bank_lanes = None
                lanes = _SLOT_DIRECT_FIXED_BANK_LANES
                initial_keys = set(_SLOT_DIRECT_PREALLOCATED)
            if not lanes or not _SLOT_DIRECT_PREALLOCATED or _SLOT_DIRECT_READY:
                return {"ok": False, "reason": "fixed_bank_not_reserved"}
            initial_layer = normalized_specs[0][0]

        built: dict[int, dict[str, Any]] = {}
        with _EXPLICIT_REGION_LOCK:
            registry_records = dict(_EXPLICIT_REGION_REGISTRY.regions)

        for ordinal, (layer_id, groups) in enumerate(normalized_specs):
            selected_lanes = (
                bank_lanes[ordinal % 2]
                if dual_bank_event and bank_lanes is not None
                else lanes
            )
            seen_experts: set[int] = set()
            keys: list[tuple[int, int, str]] = []
            for expert_id, region_ids in groups:
                if expert_id in seen_experts or not region_ids:
                    return {
                        "ok": False,
                        "reason": "invalid_registered_plan_groups",
                        "layer_id": layer_id,
                    }
                seen_experts.add(expert_id)
                keys.extend(
                    _slot_direct_binding_key(layer_id, expert_id, region_id)
                    for region_id in region_ids
                )
            if len(set(keys)) != len(keys) or len(keys) != len(selected_lanes):
                return {
                    "ok": False,
                    "reason": "registered_plan_lane_count_mismatch",
                    "layer_id": layer_id,
                    "keys": len(keys),
                    "lanes": len(selected_lanes),
                }

            missing = [key[2] for key in keys if key[2] not in registry_records]
            if missing:
                return {
                    "ok": False,
                    "reason": "registered_plan_region_missing",
                    "layer_id": layer_id,
                    "missing": missing,
                }

            keys.sort(
                key=lambda key: (int(registry_records[key[2]].length), key)
            )
            entries: list[tuple[int, ...]] = []
            lane_bindings: list[
                tuple[tuple[int, int, str], int, int, int]
            ] = []
            with _SLOT_DIRECT_LOCK:
                for key, (
                    driver_layer,
                    slot_id,
                    slot_bytes,
                ) in zip(keys, selected_lanes):
                    record = registry_records[key[2]]
                    if int(record.length) != int(slot_bytes):
                        return {
                            "ok": False,
                            "reason": "registered_plan_slot_size_mismatch",
                            "layer_id": layer_id,
                        }
                    driver_owner = _slot_direct_driver_owner_token_locked(
                        layer_id, key[1]
                    )
                    entries.append(
                        (
                            int(record.base),
                            int(record.length),
                            int(driver_layer),
                            int(driver_owner),
                            int(slot_id),
                            0,
                            _SLOT_DIRECT_NO_EVICT_FLAG
                            | UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
                        )
                    )
                    lane_bindings.append(
                        (
                            key,
                            int(driver_layer),
                            int(slot_id),
                            int(slot_bytes),
                        )
                    )

            transition_plan_id = (int(layer_id) << 2) | 2
            built[layer_id] = {
                "layer_id": layer_id,
                "expert_ids": tuple(sorted(seen_experts)),
                "keys": tuple(keys),
                "entries": tuple(entries),
                "lane_bindings": tuple(lane_bindings),
                "transition_plan_id": transition_plan_id,
                "requested_bytes": sum(entry[1] for entry in entries),
                "ordinal": ordinal,
                "bank_id": ordinal % 2 if dual_bank_event else 0,
            }

        ordered_layers = tuple(sorted(built))
        for ordinal, layer_id in enumerate(ordered_layers):
            built[layer_id]["next_layer"] = ordered_layers[
                (ordinal + 1) % len(ordered_layers)
            ]

        runtime_candidate: dict[str, Any] | None = None
        if bank_level_commit:
            assert bank_keys is not None
            bank_owner_layers: list[int] = []
            bank_expert_counts: list[int] = []
            with _SLOT_DIRECT_LOCK:
                for bank_id, keys_for_bank in enumerate(bank_keys):
                    owner_layers = {int(key[0]) for key in keys_for_bank}
                    if len(owner_layers) != 1:
                        return {
                            "ok": False,
                            "reason": "bank_level_commit_owner_ambiguous",
                            "bank_id": bank_id,
                        }
                    owner_layer = owner_layers.pop()
                    owner_plan = built.get(owner_layer)
                    if (
                        owner_plan is None
                        or int(owner_plan["bank_id"]) != bank_id
                        or frozenset(owner_plan["keys"]) != keys_for_bank
                    ):
                        return {
                            "ok": False,
                            "reason": "bank_level_commit_owner_mismatch",
                            "bank_id": bank_id,
                            "owner_layer": owner_layer,
                        }
                    bank_owner_layers.append(owner_layer)
                    bank_expert_counts.append(len(owner_plan["expert_ids"]))

                runtime_candidate = {
                    "allocated_bytes": sum(
                        int(value) for value in _SLOT_DIRECT_ALLOCATION_BYTES.values()
                    ),
                    "allocated_slots": len(_SLOT_DIRECT_PREALLOCATED),
                    "ready_slots": 0,
                    "resident_groups": sum(bank_expert_counts),
                    "bank_owner_layers": bank_owner_layers,
                    "bank_expert_counts": bank_expert_counts,
                    "active_bank": None,
                    "active_layer": None,
                    "generation": None,
                    "initial_layer": int(initial_layer),
                }

        if set(built[initial_layer]["keys"]) != initial_keys:
            return {"ok": False, "reason": "initial_plan_owner_mismatch"}

        register_results: list[dict[str, object]] = []
        rollback_results: list[dict[str, object]] = []

        def rollback_registered_plans(generation: int) -> bool:
            """Remove every plan accepted before a later registration failed."""
            for registered in reversed(register_results):
                if not bool(registered.get("ok")):
                    continue
                rollback_results.append(
                    driver_slot_destroy_plan(
                        int(registered["plan_id"]),
                        int(generation),
                        device=int(device),
                    )
                )
            return all(bool(result.get("ok")) for result in rollback_results)

        initial_plan_id = (int(initial_layer) << 2) | 1
        initial_entries = [
            (*entry[:6], _SLOT_DIRECT_NO_EVICT_FLAG)
            for entry in built[initial_layer]["entries"]
        ]
        initial_result = driver_slot_register_plan(
            initial_plan_id,
            initial_layer,
            initial_entries,
            transition=False,
            device=int(device),
        )
        register_results.append(initial_result)
        if not bool(initial_result.get("ok")):
            return {
                "ok": False,
                "reason": "initial_plan_register_failed",
                "register_results": register_results,
            }

        generation = int(initial_result["generation"])
        for layer_id in sorted(built):
            plan = built[layer_id]
            result = driver_slot_register_plan(
                int(plan["transition_plan_id"]),
                layer_id,
                plan["entries"],
                transition=True,
                device=int(device),
            )
            register_results.append(result)
            if not bool(result.get("ok")) or int(result.get("generation", -1)) != generation:
                rollback_ok = rollback_registered_plans(generation)
                return {
                    "ok": False,
                    "reason": (
                        "transition_plan_register_failed"
                        if rollback_ok
                        else "transition_plan_register_rollback_failed"
                    ),
                    "layer_id": layer_id,
                    "register_results": register_results,
                    "rollback_results": rollback_results,
                }

        with _SLOT_DIRECT_LOCK:
            _SLOT_DIRECT_REGISTERED_PLANS.clear()
            _SLOT_DIRECT_REGISTERED_PLANS.update(built)
            _SLOT_DIRECT_INITIAL_PLAN_ID = initial_plan_id
            _SLOT_DIRECT_PLAN_GENERATION = generation
            _SLOT_DIRECT_PLAN_ACTIVE_LAYER = None
            if runtime_candidate is not None:
                runtime_candidate["generation"] = generation
                _SLOT_DIRECT_REGISTERED_RUNTIME = runtime_candidate
            else:
                _SLOT_DIRECT_REGISTERED_RUNTIME = None

        result = {
            "ok": True,
            "reason": "ok",
            "initial_layer": initial_layer,
            "initial_plan_id": initial_plan_id,
            "generation": generation,
            "layer_count": len(built),
            "lane_count": len(lanes),
            "bank_count": 2 if dual_bank_event else 1,
            "dual_bank_event": dual_bank_event,
            "bank_level_commit": bank_level_commit,
            "requested_bytes_per_layer": {
                str(layer_id): int(plan["requested_bytes"])
                for layer_id, plan in built.items()
            },
            "register_results": register_results,
            "rollback_results": rollback_results,
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        return result


@_trace_slot_direct_operation("restore_layer")
def restore_uvm_registered_layer_plan(
    *,
    layer_id: int,
    expert_ids: Sequence[int],
    device: int = 0,
) -> dict[str, Any]:
    """Execute a registered full-layer plan with one compact ioctl."""
    global _SLOT_DIRECT_PLAN_GENERATION, _SLOT_DIRECT_PLAN_ACTIVE_LAYER
    global _SLOT_DIRECT_POISON_REASON
    global _SLOT_DIRECT_REGISTERED_RUNTIME

    started_ns = time.perf_counter_ns()
    layer_id = int(layer_id)
    requested_experts = tuple(sorted(int(value) for value in expert_ids))
    try:
        dual_bank_event = _slot_direct_dual_bank_event_enabled()
    except RuntimeError as exc:
        return {
            "ok": False,
            "reason": "dual_bank_config_invalid",
            "error": str(exc),
        }
    try:
        bank_level_commit = _slot_direct_bank_level_commit_enabled()
    except RuntimeError as exc:
        return {
            "ok": False,
            "reason": "bank_level_commit_config_invalid",
            "error": str(exc),
        }
    with _SLOT_DIRECT_OPERATION_LOCK:
        with _SLOT_DIRECT_LOCK:
            plan = _SLOT_DIRECT_REGISTERED_PLANS.get(layer_id)
            generation = _SLOT_DIRECT_PLAN_GENERATION
            active_layer = _SLOT_DIRECT_PLAN_ACTIVE_LAYER
            initial_plan_id = _SLOT_DIRECT_INITIAL_PLAN_ID
            runtime = _SLOT_DIRECT_REGISTERED_RUNTIME
            initial_cache = _slot_direct_cache_snapshot_locked()
            if _SLOT_DIRECT_POISON_REASON is not None:
                return {
                    "ok": False,
                    "reason": "slot_direct_session_poisoned",
                    "error": _SLOT_DIRECT_POISON_REASON,
                }
        if plan is None or generation is None or initial_plan_id is None:
            return {"ok": False, "reason": "registered_plan_missing"}
        if requested_experts != plan["expert_ids"]:
            return {
                "ok": False,
                "reason": "registered_plan_expert_set_mismatch",
            }

        if bank_level_commit:
            if (
                not dual_bank_event
                or runtime is None
                or runtime.get("generation") != generation
                or runtime.get("active_layer") != active_layer
            ):
                with _SLOT_DIRECT_LOCK:
                    _SLOT_DIRECT_POISON_REASON = (
                        "bank-level runtime state disagrees with registered plan state"
                    )
                return {
                    "ok": False,
                    "reason": "bank_level_runtime_mismatch",
                    "poisoned": True,
                }

        ordering_layer = active_layer
        if dual_bank_event:
            if ordering_layer is None:
                expected_layer = (
                    int(runtime["initial_layer"])
                    if bank_level_commit and runtime is not None
                    else min(_SLOT_DIRECT_REGISTERED_PLANS)
                )
            elif ordering_layer not in _SLOT_DIRECT_REGISTERED_PLANS:
                return {
                    "ok": False,
                    "reason": "dual_bank_active_layer_invalid",
                }
            else:
                expected_layer = int(
                    _SLOT_DIRECT_REGISTERED_PLANS[int(ordering_layer)]["next_layer"]
                )
            if layer_id != expected_layer:
                return {
                    "ok": False,
                    "reason": "dual_bank_layer_order_mismatch",
                    "active_layer": active_layer,
                    "expected_layer": expected_layer,
                    "requested_layer": layer_id,
                }

        use_initial = ordering_layer is None and layer_id == (
            int(runtime["initial_layer"])
            if bank_level_commit and runtime is not None
            else min(_SLOT_DIRECT_REGISTERED_PLANS)
        )
        plan_id = (
            int(initial_plan_id)
            if use_initial
            else int(plan["transition_plan_id"])
        )
        execute_result = driver_slot_execute_plan(
            plan_id,
            layer_id,
            int(generation),
            device=int(device),
        )
        new_generation = int(execute_result.get("new_generation", generation))

        if bool(execute_result.get("ok")):
            with _SLOT_DIRECT_LOCK:
                if bank_level_commit:
                    if runtime is None:
                        _SLOT_DIRECT_POISON_REASON = (
                            "bank-level runtime disappeared after successful execute"
                        )
                        raise RuntimeError(_SLOT_DIRECT_POISON_REASON)
                    bank_id = int(plan["bank_id"])
                    bank_owner_layers = runtime["bank_owner_layers"]
                    bank_expert_counts = runtime["bank_expert_counts"]
                    if (
                        bank_id not in (0, 1)
                        or runtime.get("generation") != generation
                        or runtime.get("active_layer") != active_layer
                    ):
                        _SLOT_DIRECT_POISON_REASON = (
                            "bank-level runtime changed across successful execute"
                        )
                        raise RuntimeError(_SLOT_DIRECT_POISON_REASON)

                    previous_experts = int(bank_expert_counts[bank_id])
                    current_experts = len(plan["expert_ids"])
                    bank_owner_layers[bank_id] = layer_id
                    bank_expert_counts[bank_id] = current_experts
                    runtime["resident_groups"] = (
                        int(runtime["resident_groups"])
                        - previous_experts
                        + current_experts
                    )
                    runtime["ready_slots"] = len(plan["keys"])
                    runtime["active_bank"] = bank_id
                    runtime["active_layer"] = layer_id
                    runtime["generation"] = new_generation
                elif dual_bank_event:
                    old_keys = []
                    for (
                        _key,
                        driver_layer,
                        slot_id,
                        _slot_bytes,
                    ) in plan["lane_bindings"]:
                        old_key = _SLOT_DIRECT_SLOT_OWNERS.get(
                            (int(driver_layer), int(slot_id))
                        )
                        if old_key is None:
                            _SLOT_DIRECT_POISON_REASON = (
                                "dual-bank policy lost a physical lane owner after "
                                f"successful kernel execute: layer={layer_id}"
                            )
                            raise RuntimeError(_SLOT_DIRECT_POISON_REASON)
                        old_keys.append(old_key)
                elif not bank_level_commit:
                    old_keys = list(_SLOT_DIRECT_PREALLOCATED)
                if not bank_level_commit:
                    for old_key in old_keys:
                        slot_id = _SLOT_DIRECT_BINDINGS.pop(old_key, None)
                        driver_layer = _SLOT_DIRECT_DRIVER_LAYERS.pop(
                            old_key, int(old_key[0])
                        )
                        if slot_id is not None:
                            owner_identity = (int(driver_layer), int(slot_id))
                            if _SLOT_DIRECT_SLOT_OWNERS.get(owner_identity) == old_key:
                                _SLOT_DIRECT_SLOT_OWNERS.pop(owner_identity, None)
                        _SLOT_DIRECT_PREALLOCATED.discard(old_key)
                        _SLOT_DIRECT_READY.discard(old_key)
                        _SLOT_DIRECT_ALLOCATION_BYTES.pop(old_key, None)
                    _SLOT_DIRECT_READY.clear()
                    for (
                        key,
                        driver_layer,
                        slot_id,
                        slot_bytes,
                    ) in plan["lane_bindings"]:
                        _SLOT_DIRECT_BINDINGS[key] = slot_id
                        _SLOT_DIRECT_DRIVER_LAYERS[key] = driver_layer
                        _SLOT_DIRECT_SLOT_OWNERS[(driver_layer, slot_id)] = key
                        _SLOT_DIRECT_PREALLOCATED.add(key)
                        _SLOT_DIRECT_READY.add(key)
                        _SLOT_DIRECT_ALLOCATION_BYTES[key] = slot_bytes
                    for expert_id in requested_experts:
                        _slot_direct_touch_group_locked(layer_id, expert_id)
                _SLOT_DIRECT_PLAN_GENERATION = new_generation
                _SLOT_DIRECT_PLAN_ACTIVE_LAYER = layer_id
                final_cache = _slot_direct_cache_snapshot_locked()
        else:
            with _SLOT_DIRECT_LOCK:
                if (
                    bool(execute_result.get("new_generation_trusted"))
                    and new_generation > 0
                ):
                    _SLOT_DIRECT_PLAN_GENERATION = new_generation
                    if runtime is not None:
                        runtime["generation"] = new_generation
                elif not bool(execute_result.get("new_generation_trusted")):
                    # The low-level session deliberately removed its cached
                    # generation. Keep policy state consistent so cleanup uses
                    # the generation-free session-abort path instead of
                    # issuing destroy with an untrusted value.
                    _SLOT_DIRECT_PLAN_GENERATION = None
                    if runtime is not None:
                        runtime["generation"] = None
                if bool(execute_result.get("poisoned")):
                    _SLOT_DIRECT_POISON_REASON = (
                        "registered slot plan execution poisoned the fixed bank: "
                        f"layer={layer_id}, plan={plan_id}"
                    )
                final_cache = _slot_direct_cache_snapshot_locked()

        ok = bool(execute_result.get("ok"))
        # The trace mode only controls what is written to disk. Keep the
        # public result contract independent from trace configuration. The
        # bank-level success path deliberately elides per-entry results, but
        # legacy commits and every failure still return complete diagnostics.
        result_detail_enabled = not bank_level_commit or not ok
        region_ids = (
            [key[2] for key in plan["keys"]] if result_detail_enabled else []
        )
        result = {
            "ok": ok,
            "reason": "ok" if ok else "registered_plan_execute_failed",
            "error": "" if ok else str(execute_result.get("reason", "")),
            "layer_id": layer_id,
            "expert_ids": list(requested_experts),
            "expert_count": len(requested_experts),
            "expert_statuses": {
                str(expert_id): ok for expert_id in requested_experts
            }
            if result_detail_enabled
            else {},
            "failed_expert_ids": [] if ok else list(requested_experts),
            "statuses": {region_id: ok for region_id in region_ids},
            "failed_region_ids": [] if ok else region_ids,
            "logical_placement": "slot_gpu" if ok else "unknown",
            "ordinary_uvm_placement": "cpu",
            "requested_bytes": int(plan["requested_bytes"]),
            "preallocate": None,
            "restore": execute_result,
            "transition": None if use_initial else execute_result,
            "rollback": None,
            "evictions": [],
            "bank_reuse_enabled": True,
            "bank_rebind_slots": 0 if use_initial else len(plan["keys"]),
            "bank_rebind_reason": "registered_plan",
            "layer_transition_used": not use_initial,
            "layer_transition_reason": "initial_registered_plan" if use_initial else "ok",
            "registered_plan_used": True,
            "dual_bank_event": dual_bank_event,
            "bank_level_commit": bank_level_commit,
            "userspace_commit_slots": 0 if bank_level_commit else len(plan["keys"]),
            "bank_id": int(plan.get("bank_id", 0)),
            "plan_id": plan_id,
            "plan_generation": generation,
            "plan_new_generation": new_generation,
            "bank_lane_plan_cache_hits": 1,
            "bank_lane_plan_builds": 0,
            "bank_lane_plan_build_ns": 0,
            "poisoned": _SLOT_DIRECT_POISON_REASON is not None,
            "poison_reason": _SLOT_DIRECT_POISON_REASON,
            **_slot_direct_trace_cache_fields(initial_cache, final_cache),
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        return result


def _restore_uvm_expert_groups_to_slots_impl(
    expert_regions: Sequence[tuple[int, Sequence[str]]],
    *,
    layer_id: int,
    device: int,
    preallocate_only: bool,
) -> dict[str, Any]:
    global _SLOT_DIRECT_ACTIVE_DEVICE, _SLOT_DIRECT_OWNER_THREAD
    global _SLOT_DIRECT_POISON_REASON

    started_ns = time.perf_counter_ns()
    normalized_groups: list[tuple[int, list[str]]] = []
    seen_groups: set[tuple[int, int]] = set()
    seen_regions: set[str] = set()
    input_error = ""
    try:
        raw_groups = list(expert_regions)
    except TypeError:
        raw_groups = []
        input_error = "expert_regions must be a finite sequence"

    for raw_group in raw_groups:
        try:
            expert_id, raw_region_ids = raw_group
            expert_id = int(expert_id)
            requested = list(
                dict.fromkeys(str(region_id) for region_id in raw_region_ids)
            )
        except (TypeError, ValueError):
            input_error = "each expert group must contain an integer id and regions"
            break
        group = (int(layer_id), expert_id)
        if group in seen_groups:
            input_error = f"duplicate expert group: {group}"
            break
        overlap = seen_regions.intersection(requested)
        if overlap:
            input_error = f"regions have multiple expert owners: {sorted(overlap)!r}"
            break
        seen_groups.add(group)
        seen_regions.update(requested)
        normalized_groups.append((expert_id, requested))

    normalized_groups.sort(key=lambda item: item[0])
    expert_ids = [expert_id for expert_id, _region_ids in normalized_groups]
    requested_region_ids = [
        region_id
        for _expert_id, region_ids in normalized_groups
        for region_id in region_ids
    ]
    statuses = {region_id: False for region_id in requested_region_ids}
    expert_statuses = {expert_id: False for expert_id in expert_ids}
    with _SLOT_DIRECT_LOCK:
        initial_cache = _slot_direct_cache_snapshot_locked()
        initial_lane_plan_stats = _slot_direct_fixed_bank_plan_stats_locked()

    preallocate_result: dict[str, object] | None = None
    restore_result: dict[str, object] | None = None
    transition_result: dict[str, object] | None = None
    rollback: Any = None
    evictions: list[dict[str, Any]] = []
    bank_rebind_slots = 0
    bank_rebind_reason = "not_attempted"
    slot_ids_by_region: dict[str, int] = {}
    requested_bytes = 0
    layer_transition_used = False
    layer_transition_reason = "disabled"

    def finish(ok: bool, reason: str, *, error: str = "") -> dict[str, Any]:
        with _SLOT_DIRECT_LOCK:
            final_cache = _slot_direct_cache_snapshot_locked()
            final_lane_plan_stats = _slot_direct_fixed_bank_plan_stats_locked()
            poisoned = _SLOT_DIRECT_POISON_REASON is not None
            poison_reason = _SLOT_DIRECT_POISON_REASON
        result: dict[str, Any] = {
            "ok": bool(ok),
            "reason": reason,
            "error": error,
            "layer_id": int(layer_id),
            "expert_ids": expert_ids,
            "expert_count": len(expert_ids),
            "expert_statuses": {
                str(expert_id): bool(expert_statuses[expert_id])
                for expert_id in expert_ids
            },
            "failed_expert_ids": [
                expert_id
                for expert_id in expert_ids
                if not expert_statuses[expert_id]
            ],
            "statuses": statuses,
            "failed_region_ids": [
                region_id
                for region_id in requested_region_ids
                if not statuses[region_id]
            ],
            "logical_placement": (
                "slot_reserved"
                if ok and preallocate_only
                else "slot_gpu" if ok else "unknown"
            ),
            "ordinary_uvm_placement": "cpu",
            "preallocate_only": bool(preallocate_only),
            "requested_bytes": int(requested_bytes),
            "preallocate": preallocate_result,
            "restore": restore_result,
            "transition": transition_result,
            "rollback": rollback,
            "evictions": evictions,
            "bank_reuse_enabled": _slot_direct_reuse_bank_enabled(),
            "bank_rebind_slots": int(bank_rebind_slots),
            "bank_rebind_reason": bank_rebind_reason,
            "layer_transition_used": layer_transition_used,
            "layer_transition_reason": layer_transition_reason,
            "bank_lane_plan_cache_hits": (
                final_lane_plan_stats[0] - initial_lane_plan_stats[0]
            ),
            "bank_lane_plan_builds": (
                final_lane_plan_stats[1] - initial_lane_plan_stats[1]
            ),
            "bank_lane_plan_build_ns": (
                final_lane_plan_stats[2] - initial_lane_plan_stats[2]
            ),
            "slot_ids": slot_ids_by_region,
            "poisoned": poisoned,
            "poison_reason": poison_reason,
            **_slot_direct_trace_cache_fields(initial_cache, final_cache),
            "wall_ns": time.perf_counter_ns() - started_ns,
        }
        _write_slot_direct_trace(
            {
                "operation": (
                    "preallocate_layer" if preallocate_only else "restore_layer"
                ),
                **result,
            }
        )
        return result

    if input_error:
        return finish(False, "invalid_expert_groups", error=input_error)
    if not normalized_groups:
        return finish(
            not preallocate_only,
            "empty" if not preallocate_only else "empty_init_bank",
        )
    if any(not region_ids for _expert_id, region_ids in normalized_groups):
        return finish(False, "empty_expert_regions")

    with _EXPLICIT_REGION_LOCK:
        records = {
            region_id: _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            for region_id in requested_region_ids
        }
    missing = [region_id for region_id, record in records.items() if record is None]
    if missing:
        return finish(
            False,
            "unregistered_region",
            error=f"missing={missing!r}",
        )
    requested_bytes = sum(
        int(records[region_id].length)  # type: ignore[union-attr]
        for region_id in requested_region_ids
    )

    keys_by_expert = {
        expert_id: [
            _slot_direct_binding_key(layer_id, expert_id, region_id)
            for region_id in region_ids
        ]
        for expert_id, region_ids in normalized_groups
    }
    all_keys = [
        key for expert_id in expert_ids for key in keys_by_expert[expert_id]
    ]
    region_ids_by_expert = dict(normalized_groups)
    requested_groups = {(int(layer_id), expert_id) for expert_id in expert_ids}

    if preallocate_only:
        with _SLOT_DIRECT_LOCK:
            init_state, init_error = _slot_direct_preallocate_only_preflight_locked(
                all_keys,
                records,
                device=int(device),
            )
            if init_state == "idempotent":
                slot_ids_by_region.update(
                    {
                        key[2]: int(_SLOT_DIRECT_BINDINGS[key])
                        for key in all_keys
                    }
                )
        if init_state == "conflict":
            return finish(
                False,
                "preallocate_only_state_conflict",
                error=init_error,
            )
        if init_state == "idempotent":
            return finish(True, "already_layer_slot_preallocated")

    contract_error = ""
    defer_fixed_bank_bindings = False
    with _SLOT_DIRECT_LOCK:
        if _SLOT_DIRECT_POISON_REASON is not None:
            contract_error = _SLOT_DIRECT_POISON_REASON
        else:
            current_thread = threading.get_ident()
            if (
                _SLOT_DIRECT_OWNER_THREAD is not None
                and _SLOT_DIRECT_OWNER_THREAD != current_thread
            ):
                contract_error = "slot_direct_single_execution_thread_only"
            elif (
                _SLOT_DIRECT_ACTIVE_DEVICE is not None
                and _SLOT_DIRECT_ACTIVE_DEVICE != int(device)
            ):
                contract_error = "slot_direct_single_gpu_only"
            else:
                _SLOT_DIRECT_OWNER_THREAD = current_thread
                _SLOT_DIRECT_ACTIVE_DEVICE = int(device)
                defer_fixed_bank_bindings = (
                    _slot_direct_reuse_bank_enabled()
                    and _slot_direct_precomputed_lanes_enabled()
                    and _SLOT_DIRECT_FIXED_BANK_LANES is not None
                )
                for key in all_keys:
                    if (
                        key not in _SLOT_DIRECT_BINDINGS
                        and not defer_fixed_bank_bindings
                    ):
                        _slot_direct_slot_id_locked(key)

    if contract_error:
        reason = (
            "slot_direct_session_poisoned"
            if _SLOT_DIRECT_POISON_REASON is not None
            else contract_error
        )
        return finish(False, reason, error=contract_error)

    ready_experts: set[int] = set()
    pending_experts: list[int] = []
    with _SLOT_DIRECT_LOCK:
        for expert_id in expert_ids:
            ready = [key in _SLOT_DIRECT_READY for key in keys_by_expert[expert_id]]
            if any(ready) and not all(ready):
                _SLOT_DIRECT_POISON_REASON = (
                    "partial logical expert readiness before layer restore: "
                    f"layer={layer_id}, expert={expert_id}"
                )
                return finish(
                    False,
                    "partial_expert_readiness",
                    error=_SLOT_DIRECT_POISON_REASON,
                )
            if all(ready):
                ready_experts.add(expert_id)
                _slot_direct_touch_group_locked(layer_id, expert_id)
            else:
                pending_experts.append(expert_id)
        for expert_id, region_ids in normalized_groups:
            for region_id, key in zip(region_ids, keys_by_expert[expert_id]):
                slot_id = _SLOT_DIRECT_BINDINGS.get(key)
                if slot_id is not None:
                    slot_ids_by_region[region_id] = int(slot_id)

    for expert_id in ready_experts:
        expert_statuses[expert_id] = True
        for key in keys_by_expert[expert_id]:
            statuses[key[2]] = True
    if not pending_experts:
        return finish(
            True,
            (
                "already_layer_slot_preallocated"
                if preallocate_only
                else "already_layer_slot_ready"
            ),
        )

    pending_keys = [
        key for expert_id in pending_experts for key in keys_by_expert[expert_id]
    ]
    entries_by_key: dict[tuple[int, int, str], tuple[int, ...]] = {}
    with _SLOT_DIRECT_LOCK:
        for expert_id, region_ids in normalized_groups:
            driver_owner = _slot_direct_driver_owner_token_locked(layer_id, expert_id)
            for region_id, key in zip(region_ids, keys_by_expert[expert_id]):
                record = records[region_id]
                assert record is not None
                if key in _SLOT_DIRECT_BINDINGS:
                    driver_layer, driver_slot = (
                        _slot_direct_driver_identity_locked(key)
                    )
                else:
                    # Capacity selection and bank matching only consume the
                    # geometry at this stage. Avoid allocating 1,152 hashed
                    # provisional slot ids which a complete rebind would
                    # immediately discard.
                    driver_layer = _SLOT_DIRECT_REUSE_BANK_LAYER
                    driver_slot = 0
                entries_by_key[key] = (
                    int(record.base),
                    int(record.length),
                    int(driver_layer),
                    int(driver_owner),
                    int(driver_slot),
                    0,
                    _SLOT_DIRECT_NO_EVICT_FLAG,
                )
        victims, capacity_error = _slot_direct_capacity_victims_for_groups_locked(
            all_keys,
            entries_by_key,
            requested_groups,
            scope_name="one layer working set",
        )
        rebind_plan: dict[
            tuple[int, int, str],
            tuple[tuple[int, int, str], int, int, int],
        ] = {}
        if not capacity_error:
            rebind_plan, rebind_evictions, bank_rebind_reason = (
                _slot_direct_rebind_capacity_victims_locked(
                    pending_keys,
                    entries_by_key,
                    victims,
                )
            )
            evictions.extend(rebind_evictions)
        rebound_keys = set(rebind_plan)
        bank_rebind_slots = len(rebound_keys)

    if capacity_error:
        return finish(False, "slot_cache_capacity_failed", error=capacity_error)

    # A complete compatible transfer needs no mode 19. The legacy release path
    # remains available for non-reuse configurations and shape mismatches.
    release_victims = [] if rebound_keys else victims
    for victim_layer, victim_expert, victim_region_ids in release_victims:
        release = release_uvm_regions_from_slots(
            victim_region_ids,
            layer_id=victim_layer,
            expert_id=victim_expert,
            device=int(device),
            trigger="capacity",
        )
        evictions.append(
            {
                "layer_id": victim_layer,
                "expert_id": victim_expert,
                "region_ids": victim_region_ids,
                "slot_ids": dict(release.get("slot_ids", {})),
                "release": release,
            }
        )
        if not bool(release.get("ok")):
            return finish(False, "slot_cache_eviction_failed")

    if defer_fixed_bank_bindings and len(rebound_keys) != len(pending_keys):
        # The cached fast path is deliberately all-or-nothing. A shape or
        # capacity mismatch falls back to the audited allocation path only
        # after old victims have been released.
        with _SLOT_DIRECT_LOCK:
            for key in pending_keys:
                if key not in rebound_keys and key not in _SLOT_DIRECT_BINDINGS:
                    _slot_direct_slot_id_locked(key)

    preallocate_entries_by_key: dict[
        tuple[int, int, str], tuple[int, ...]
    ] = {}
    with _SLOT_DIRECT_LOCK:
        for expert_id in pending_experts:
            driver_owner = _slot_direct_driver_owner_token_locked(layer_id, expert_id)
            region_ids = region_ids_by_expert[expert_id]
            for region_id, key in zip(region_ids, keys_by_expert[expert_id]):
                record = records[region_id]
                assert record is not None
                if key in rebind_plan:
                    (
                        _old_key,
                        driver_layer,
                        driver_slot,
                        _bytes,
                    ) = rebind_plan[key]
                    slot_ids_by_region[region_id] = int(driver_slot)
                else:
                    driver_layer, driver_slot = _slot_direct_driver_identity_locked(key)
                entry = (
                    int(record.base),
                    int(record.length),
                    int(driver_layer),
                    int(driver_owner),
                    int(driver_slot),
                    0,
                    _SLOT_DIRECT_NO_EVICT_FLAG,
                )
                entries_by_key[key] = entry
                preallocate_entries_by_key[key] = (
                    *entry[:-1],
                    entry[-1]
                    | (
                        UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
                        if key in rebound_keys
                        else 0
                    ),
                )
        needs_preallocate = [
            key for key in pending_keys if key not in _SLOT_DIRECT_PREALLOCATED
        ]

    try:
        transition_enabled = _slot_direct_layer_transition_enabled()
    except RuntimeError as exc:
        return finish(False, "layer_transition_config_invalid", error=str(exc))

    complete_fixed_bank_rebind = (
        bool(pending_keys)
        and len(rebound_keys) == len(pending_keys)
        and len(rebind_plan) == len(pending_keys)
        and set(needs_preallocate) == set(pending_keys)
        and set(pending_keys) == set(all_keys)
    )
    use_layer_transition = transition_enabled and complete_fixed_bank_rebind
    if not transition_enabled:
        layer_transition_reason = "disabled"
    elif not rebound_keys:
        layer_transition_reason = "initial_bank_allocation"
    elif not complete_fixed_bank_rebind:
        layer_transition_reason = "incomplete_fixed_bank_rebind"

    if use_layer_transition:
        layer_transition_used = True
        layer_transition_reason = "executing"
        try:
            transition_result = driver_slot_transition_ranges(
                [preallocate_entries_by_key[key] for key in pending_keys],
                device=int(device),
            )
        except Exception as exc:
            transition_result = {
                "ok": False,
                "reason": "slot_layer_transition_exception",
                "error": repr(exc),
                "entry_statuses": [None] * len(pending_keys),
                "owner_cutover": None,
                "owner_cutover_trusted": False,
                "owner_state_indeterminate": True,
            }

        cutover_trusted = bool(
            transition_result.get("owner_cutover_trusted", False)
        )
        owner_cutover = transition_result.get("owner_cutover")
        if cutover_trusted and owner_cutover is True:
            try:
                with _SLOT_DIRECT_LOCK:
                    _slot_direct_commit_rebinds_locked(
                        rebind_plan,
                        set(pending_keys),
                    )
                    _SLOT_DIRECT_PREALLOCATED.update(pending_keys)
                    _SLOT_DIRECT_ALLOCATION_BYTES.update(
                        {
                            key: int(entries_by_key[key][1])
                            for key in pending_keys
                        }
                    )
            except Exception as exc:
                with _SLOT_DIRECT_LOCK:
                    _SLOT_DIRECT_POISON_REASON = (
                        "mode20 owner cutover could not be committed in policy: "
                        f"layer={layer_id}, error={exc!r}"
                    )
                layer_transition_reason = "policy_owner_commit_failed"
                return finish(
                    False,
                    "slot_layer_transition_failed",
                    error=_SLOT_DIRECT_POISON_REASON or repr(exc),
                )

        if not bool(transition_result.get("ok")):
            with _SLOT_DIRECT_LOCK:
                if cutover_trusted and owner_cutover is True:
                    _SLOT_DIRECT_POISON_REASON = (
                        "mode20 copy/map failed after fixed-bank owner cutover: "
                        f"layer={layer_id}"
                    )
                    layer_transition_reason = "post_cutover_failure"
                elif cutover_trusted and owner_cutover is False:
                    for old_key, _driver_layer, _slot_id, _bytes in (
                        rebind_plan.values()
                    ):
                        _SLOT_DIRECT_READY.discard(old_key)
                    _SLOT_DIRECT_POISON_REASON = (
                        "mode20 failed before fixed-bank owner cutover: "
                        f"layer={layer_id}"
                    )
                    layer_transition_reason = "pre_cutover_failure"
                else:
                    for old_key, _driver_layer, _slot_id, _bytes in (
                        rebind_plan.values()
                    ):
                        _SLOT_DIRECT_READY.discard(old_key)
                    _SLOT_DIRECT_POISON_REASON = (
                        "mode20 returned indeterminate fixed-bank ownership: "
                        f"layer={layer_id}"
                    )
                    layer_transition_reason = "indeterminate_owner"
            return finish(False, "slot_layer_transition_failed")

        if not cutover_trusted or owner_cutover is not True:
            with _SLOT_DIRECT_LOCK:
                _SLOT_DIRECT_POISON_REASON = (
                    "mode20 success did not prove full owner cutover: "
                    f"layer={layer_id}"
                )
            layer_transition_reason = "unproven_success_cutover"
            return finish(False, "slot_layer_transition_failed")

        restore_result = transition_result
        layer_transition_reason = "ok"

    if needs_preallocate and not use_layer_transition:
        preallocate_raised = False
        try:
            preallocate_result = driver_slot_preallocate_ranges(
                [preallocate_entries_by_key[key] for key in needs_preallocate],
                device=int(device),
            )
        except Exception as exc:
            preallocate_raised = True
            preallocate_result = {
                "ok": False,
                "reason": "slot_preallocate_exception",
                "error": repr(exc),
                "entry_statuses": [None] * len(needs_preallocate),
            }
        raw_statuses = preallocate_result.get("entry_statuses")
        if isinstance(raw_statuses, list) and len(raw_statuses) == len(
            needs_preallocate
        ):
            successful_preallocations = {
                key
                for key, status in zip(needs_preallocate, raw_statuses)
                if type(status) is int and status == 0
            }
        elif bool(preallocate_result.get("ok")):
            successful_preallocations = set(needs_preallocate)
        else:
            successful_preallocations = set()
        with _SLOT_DIRECT_LOCK:
            successful_rebinds = successful_preallocations & rebound_keys
            _slot_direct_commit_rebinds_locked(rebind_plan, successful_rebinds)
            _SLOT_DIRECT_PREALLOCATED.update(successful_preallocations)
            _SLOT_DIRECT_ALLOCATION_BYTES.update(
                {
                    key: int(entries_by_key[key][1])
                    for key in successful_preallocations
                }
            )
            if bool(preallocate_result.get("ok")):
                _slot_direct_capture_fixed_bank_lanes_locked(all_keys)
        if not bool(preallocate_result.get("ok")):
            if rebound_keys:
                with _SLOT_DIRECT_LOCK:
                    _SLOT_DIRECT_POISON_REASON = (
                        "mode8 layer bank rebind did not complete every slot: "
                        f"layer={layer_id}"
                    )
            elif successful_preallocations:
                rollback_results = []
                for expert_id in pending_experts:
                    successful_regions = [
                        key[2]
                        for key in keys_by_expert[expert_id]
                        if key in successful_preallocations
                    ]
                    if successful_regions:
                        rollback_results.append(
                            release_uvm_regions_from_slots(
                                successful_regions,
                                layer_id=layer_id,
                                expert_id=expert_id,
                                device=int(device),
                                trigger="explicit",
                            )
                        )
                rollback = rollback_results
            elif preallocate_raised:
                with _SLOT_DIRECT_LOCK:
                    _SLOT_DIRECT_POISON_REASON = (
                        "mode8 layer rebind failed with indeterminate ownership: "
                        f"layer={layer_id}"
                    )
            return finish(False, "slot_preallocate_failed")

    if preallocate_only:
        # A warmup reservation must never evict/rebind an active bank or make
        # weights GPU-visible.  The initialization caller selects an empty
        # canonical layer; fail closed if that contract changes.
        if rebound_keys or use_layer_transition:
            with _SLOT_DIRECT_LOCK:
                _SLOT_DIRECT_POISON_REASON = (
                    "initial fixed-bank reservation attempted an owner cutover: "
                    f"layer={layer_id}, rebound={len(rebound_keys)}"
                )
            return finish(
                False,
                "preallocate_only_owner_cutover_forbidden",
                error=_SLOT_DIRECT_POISON_REASON,
            )
        return finish(True, "preallocated_only")

    if not use_layer_transition:
        try:
            restore_result = driver_slot_restore_ranges(
                [entries_by_key[key] for key in pending_keys],
                device=int(device),
            )
        except Exception as exc:
            restore_result = {
                "ok": False,
                "reason": "slot_restore_exception",
                "error": repr(exc),
                "entry_statuses": [None] * len(pending_keys),
            }

    if restore_result is not None and bool(restore_result.get("ok")):
        with _SLOT_DIRECT_LOCK:
            _SLOT_DIRECT_READY.update(pending_keys)
            for expert_id in pending_experts:
                _slot_direct_touch_group_locked(layer_id, expert_id)
        for expert_id in pending_experts:
            expert_statuses[expert_id] = True
            for key in keys_by_expert[expert_id]:
                statuses[key[2]] = True
        return finish(True, "ok")

    if rebound_keys:
        with _SLOT_DIRECT_LOCK:
            _SLOT_DIRECT_POISON_REASON = (
                "mode10 failed after fixed-bank layer rebind: "
                f"layer={layer_id}"
            )
    else:
        rollback_results = []
        for expert_id in pending_experts:
            rollback_results.append(
                release_uvm_regions_from_slots(
                    region_ids_by_expert[expert_id],
                    layer_id=layer_id,
                    expert_id=expert_id,
                    device=int(device),
                    trigger="explicit",
                )
            )
        rollback = rollback_results
    return finish(False, "slot_restore_failed")


def _merge_overlapping_uvm_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping positive ranges without crossing allocation borders.

    Two independent ``cudaMallocManaged`` allocations can be virtually
    adjacent.  CUDA rejects one prefetch spanning that shared boundary, so
    touching ranges must remain separate unless allocation identity is known.
    """
    normalized = sorted(
        (int(base), int(length))
        for base, length in ranges
        if int(base) > 0 and int(length) > 0
    )
    if not normalized:
        return []

    merged: list[tuple[int, int]] = []
    current_base, current_length = normalized[0]
    current_end = current_base + current_length
    for base, length in normalized[1:]:
        end = base + length
        if base < current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_base, current_end - current_base))
        current_base = base
        current_end = end
    merged.append((current_base, current_end - current_base))
    return merged


def _set_uvm_region_state_locked(
    regions: list[KernelRegion],
    state: RegionState,
) -> bool:
    update_ok = True
    for region in regions:
        try:
            result = driver_update_region_state(
                region_id=_stable_driver_region_id(region.region_id),
                state=_driver_region_state_id(state.value),
            )
            status = 0 if result is None else int(result)
        except Exception:
            logger.warning(
                "Explicit UVM driver state update raised: region_id=%s state=%s",
                region.region_id,
                state.value,
                exc_info=True,
            )
            update_ok = False
            break
        if status < 0:
            logger.warning(
                "Explicit UVM driver state update failed: "
                "region_id=%s state=%s status=%s",
                region.region_id,
                state.value,
                status,
            )
            update_ok = False
            break

    if update_ok:
        for region in regions:
            region.state = state
        return True

    # A partially updated batch cannot support a placement claim. Degrade the
    # complete application batch first, then converge every driver entry to
    # UNKNOWN on a best-effort basis.
    for region in regions:
        region.state = RegionState.UNKNOWN
    for region in regions:
        try:
            driver_update_region_state(
                region_id=_stable_driver_region_id(region.region_id),
                state=_driver_region_state_id(RegionState.UNKNOWN.value),
            )
        except Exception:
            logger.debug(
                "Failed to roll explicit UVM driver state back to UNKNOWN: %s",
                region.region_id,
                exc_info=True,
            )
    return False


def prepare_uvm_regions_for_gpu_placement(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    executor: str = "auto",
) -> dict[str, Any]:
    """Submit one coalesced GPU prefetch batch without committing GPU state.

    A successful return leaves every named region in ``PREFETCHING_GPU``.
    The caller must synchronize CUDA and then call
    :func:`commit_uvm_regions_gpu_placement`.  Any validation, submission, or
    partial-batch failure atomically degrades all present regions to UNKNOWN.
    """
    requested = list(dict.fromkeys(str(region_id) for region_id in region_ids))
    if not requested:
        return {"ok": True, "reason": "no_regions"}

    with _EXPLICIT_REGION_LOCK:
        regions = [
            region
            for region_id in requested
            if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id)) is not None
        ]
        missing = [
            region_id
            for region_id in requested
            if region_id not in _EXPLICIT_REGION_REGISTRY.regions
        ]
        if missing:
            _set_uvm_region_state_locked(regions, RegionState.UNKNOWN)
            return {"ok": False, "reason": "unregistered_regions"}
        ranges = _merge_overlapping_uvm_ranges(
            [(region.base, region.length) for region in regions]
        )
        if not ranges:
            _set_uvm_region_state_locked(regions, RegionState.UNKNOWN)
            return {"ok": False, "reason": "no_valid_ranges"}
        if not _set_uvm_region_state_locked(
            regions,
            RegionState.PREFETCHING_GPU,
        ):
            return {"ok": False, "reason": "driver_state_update_failed"}

    executor_used = "not_submitted"
    driver_result = 0
    try:
        successes, executor_used, driver_result = (
            batch_prefetch_ranges_to_device_with_executor(
                ranges,
                device,
                executor=executor,
            )
        )
    except Exception:
        logger.warning("Failed to submit final Hot Expert GPU placement", exc_info=True)
        successes = 0

    complete = int(successes) == len(ranges)
    if not complete:
        logger.error(
            "Explicit Hot Expert prefetch incomplete: requested_regions=%d "
            "ranges=%d successes=%d executor_requested=%s "
            "executor_used=%s driver_result=%d",
            len(requested),
            len(ranges),
            int(successes),
            executor,
            executor_used,
            int(driver_result),
        )
        with _EXPLICIT_REGION_LOCK:
            current = [
                region
                for region_id in requested
                if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id))
                is not None
            ]
            _set_uvm_region_state_locked(current, RegionState.UNKNOWN)

    return {
        "ok": complete,
        "reason": "prefetch_submitted" if complete else "prefetch_incomplete",
    }


def commit_uvm_regions_gpu_placement(
    region_ids: list[str] | tuple[str, ...],
    *,
    synchronized: bool,
) -> dict[str, Any]:
    """Commit GPU only after the caller's single CUDA synchronization."""
    requested = list(dict.fromkeys(str(region_id) for region_id in region_ids))
    with _EXPLICIT_REGION_LOCK:
        regions = [
            region
            for region_id in requested
            if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id)) is not None
        ]
        missing = [
            region_id
            for region_id in requested
            if region_id not in _EXPLICIT_REGION_REGISTRY.regions
        ]
        ready = (
            bool(requested)
            and not missing
            and all(region.state == RegionState.PREFETCHING_GPU for region in regions)
        )
        committed = bool(synchronized) and ready
        reason = "placement_not_synchronized"
        if committed:
            committed = _set_uvm_region_state_locked(
                regions,
                RegionState.GPU,
            )
            reason = "committed" if committed else "driver_state_update_failed"
        else:
            _set_uvm_region_state_locked(regions, RegionState.UNKNOWN)

    return {"ok": committed, "reason": reason}


def query_uvm_region_states(
    region_ids: list[str] | tuple[str, ...],
) -> dict[str, str]:
    with _EXPLICIT_REGION_LOCK:
        return {
            region_id: region.state.value
            for region_id in (str(value) for value in region_ids)
            if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id)) is not None
        }


def snapshot_uvm_region_registry(
    region_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Return immutable JSON-ready geometry and application registry states."""
    requested = list(dict.fromkeys(str(value) for value in region_ids))
    with _EXPLICIT_REGION_LOCK:
        records = []
        missing = []
        for region_id in requested:
            region = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            if region is None:
                missing.append(region_id)
                continue
            records.append(
                {
                    "region_id": region.region_id,
                    "driver_region_id": _stable_driver_region_id(region.region_id),
                    "base": int(region.base),
                    "length": int(region.length),
                    "state": region.state.value,
                }
            )
    return {
        "ok": not missing and len(records) == len(requested),
        "requested_region_count": len(requested),
        "regions": records,
        "missing_region_ids": missing,
    }


def query_uvm_regions_physical_residency(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    wait_for_trackers: bool = False,
) -> dict[str, Any]:
    """Query real page residency for explicitly registered region geometry."""
    snapshot = snapshot_uvm_region_registry(region_ids)
    regions = snapshot["regions"]
    assert isinstance(regions, list)
    ranges = [
        (int(record["base"]), int(record["length"]))
        for record in regions
        if isinstance(record, dict)
    ]
    expected_bytes = sum(length for _, length in ranges)
    if snapshot["ok"] is not True:
        return {
            "ok": False,
            "reason": "registry_snapshot_incomplete",
            "expected_bytes": expected_bytes,
            "registry": snapshot,
        }
    result = driver_batch_query_ranges_residency_bytes(
        ranges,
        device=device,
        wait_for_trackers=wait_for_trackers,
    )
    return {
        **result,
        "reason": "queried" if result.get("ok") else "driver_query_failed",
        "region_count": len(ranges),
        "expected_bytes": expected_bytes,
        "wait_for_trackers": bool(wait_for_trackers),
    }


def reconcile_uvm_regions_cpu_state_from_physical_residency(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    wait_for_trackers: bool = False,
) -> dict[str, Any]:
    """Commit CPU state only after proving every named byte is CPU-resident.

    Explicit-region registration deliberately starts in ``UNKNOWN``.  A
    destroy/rebuild cycle can therefore lose the application/driver placement
    tag even though the managed pages at the stable VA remain CPU-resident.
    Re-running the GPU-chunk release primitive on those pages is invalid.  This
    helper restores only the lost tag, and only when the physical residency
    query proves complete CPU residency with no GPU or unknown bytes.
    """
    requested = list(dict.fromkeys(str(value) for value in region_ids))
    if not requested:
        return {
            "ok": True,
            "reason": "no_regions",
            "physical_cpu_only": True,
            "reconciled": False,
            "region_count": 0,
            "expected_bytes": 0,
        }

    with _EXPLICIT_REGION_LOCK:
        records = [
            _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            for region_id in requested
        ]
        if any(region is None for region in records):
            return {
                "ok": False,
                "reason": "unregistered_regions",
                "physical_cpu_only": False,
                "reconciled": False,
                "missing_region_ids": [
                    region_id
                    for region_id, region in zip(requested, records)
                    if region is None
                ],
            }
        regions = [region for region in records if region is not None]
        geometry = tuple((int(region.base), int(region.length)) for region in regions)

    expected_bytes = sum(length for _, length in geometry)
    physical = dict(
        driver_batch_query_ranges_residency_bytes(
            list(geometry),
            device=int(device),
            wait_for_trackers=bool(wait_for_trackers),
        )
    )
    physical_cpu_only = bool(
        physical.get("ok") is True
        and int(physical.get("queried_bytes", -1)) == expected_bytes
        and int(physical.get("cpu_bytes", -1)) == expected_bytes
        and int(physical.get("gpu_bytes", -1)) == 0
        and int(physical.get("any_bytes", -1)) == expected_bytes
        and int(physical.get("unknown_bytes", -1)) == 0
    )
    if not physical_cpu_only:
        return {
            "ok": bool(physical.get("ok") is True),
            "reason": (
                "physical_not_cpu_only"
                if physical.get("ok") is True
                else "driver_query_failed"
            ),
            "physical_cpu_only": False,
            "reconciled": False,
            "region_count": len(regions),
            "expected_bytes": expected_bytes,
            "physical_residency": physical,
        }

    with _EXPLICIT_REGION_LOCK:
        current = [
            _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            for region_id in requested
        ]
        if any(before is not after for before, after in zip(regions, current)):
            return {
                "ok": False,
                "reason": "registry_changed_after_physical_audit",
                "physical_cpu_only": True,
                "reconciled": False,
                "region_count": len(regions),
                "expected_bytes": expected_bytes,
                "physical_residency": physical,
            }
        state_update_ok = _set_uvm_region_state_locked(regions, RegionState.CPU)

    return {
        "ok": bool(state_update_ok),
        "reason": "reconciled" if state_update_ok else "driver_state_update_failed",
        "physical_cpu_only": True,
        "reconciled": bool(state_update_ok),
        "region_count": len(regions),
        "expected_bytes": expected_bytes,
        "physical_residency": physical,
    }
