# SPDX-License-Identifier: Apache-2.0
"""Application-side explicit UVM region model.

The driver deliberately has no knowledge of pools, layers, experts, roles or
eviction policy.  This module owns those bits of meaning when they are useful
to the scheduler and presents the driver with only ``region_id`` and VA
geometry.  It is intentionally dependency-free so it can also be used by unit
tests and CPU-only startup paths.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


class RegionState(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    PREFETCHING_GPU = "prefetching_gpu"
    PREFETCHING_CPU = "prefetching_cpu"
    UNKNOWN = "unknown"


@dataclass
class KernelRegion:
    """The complete region metadata that may be sent to the kernel."""

    region_id: str
    base: int
    length: int
    state: RegionState = RegionState.UNKNOWN

    def __post_init__(self) -> None:
        self.region_id = str(self.region_id)
        self.base = int(self.base)
        self.length = int(self.length)
        if not self.region_id or self.base <= 0 or self.length <= 0:
            raise ValueError("region_id, base and length must be positive")
        self.state = normalize_region_state(self.state)


@dataclass
class ExpertRecord:
    """Application-level ownership and aggregate residency for one expert."""

    layer_id: int
    global_expert_id: int
    aggregate_state: RegionState = RegionState.UNKNOWN
    envelope_region_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.layer_id = int(self.layer_id)
        self.global_expert_id = int(self.global_expert_id)
        self.aggregate_state = normalize_region_state(self.aggregate_state)
        self.envelope_region_ids = [str(value) for value in self.envelope_region_ids]

    @property
    def key(self) -> tuple[int, int]:
        return self.layer_id, self.global_expert_id


@dataclass(frozen=True)
class PhysicalKVSpan:
    """Application-only mapping from block coordinates to a VA span."""

    span_id: str
    ptr: int
    size_bytes: int
    page_size: int
    max_block_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "span_id", str(self.span_id))
        object.__setattr__(self, "ptr", int(self.ptr))
        object.__setattr__(self, "size_bytes", int(self.size_bytes))
        object.__setattr__(self, "page_size", int(self.page_size))
        object.__setattr__(self, "max_block_id", int(self.max_block_id))
        if not self.span_id or self.ptr <= 0:
            raise ValueError("span_id and ptr are required")
        if self.size_bytes <= 0 or self.page_size <= 0:
            raise ValueError("size_bytes and page_size must be positive")
        if self.max_block_id < 0:
            raise ValueError("max_block_id must be non-negative")

    def ranges_for_blocks(self, block_ids: Iterable[int]) -> list[tuple[int, int]]:
        ids = sorted({int(block_id) for block_id in block_ids})
        # Block 0 is the permanently reserved null block. Runtime pool resizing
        # never requests it, but initialization-time convergence must be able
        # to rematerialize it after memory pressure evicts its physical page.
        ids = [block_id for block_id in ids if 0 <= block_id <= self.max_block_id]
        if not ids:
            return []
        ranges: list[tuple[int, int]] = []
        first = previous = ids[0]
        for block_id in ids[1:]:
            if block_id == previous + 1:
                previous = block_id
                continue
            ranges.append(self._range(first, previous))
            first = previous = block_id
        ranges.append(self._range(first, previous))
        return ranges

    def _range(self, first_block: int, last_block: int) -> tuple[int, int]:
        return (
            int(self.ptr) + int(first_block) * int(self.page_size),
            (int(last_block) - int(first_block) + 1) * int(self.page_size),
        )


@dataclass
class LogicalKVBlock:
    group_id: int
    block_id: int
    state: RegionState = RegionState.UNKNOWN
    lifecycle_state: str | None = None
    ref_cnt: int | None = None

    def __post_init__(self) -> None:
        self.group_id = int(self.group_id)
        self.block_id = int(self.block_id)
        self.state = normalize_region_state(self.state)


def normalize_region_state(state: RegionState | str | None) -> RegionState:
    if isinstance(state, RegionState):
        return state
    value = str(state or "unknown").strip().lower()
    if value == "released":
        value = "cpu"
    try:
        return RegionState(value)
    except ValueError:
        return RegionState.UNKNOWN


def compile_expert_envelopes(
    *,
    layer_id: int,
    global_expert_id: int,
    slices: Sequence[tuple[int, int]],
    envelope_bytes: int,
    existing_owners: MutableMapping[str, tuple[int, int]] | None = None,
) -> ExpertRecord:
    """Compile an expert's slice geometry into exclusive envelope IDs.

    ``existing_owners`` is shared by the caller across experts.  A geometry
    collision is rejected instead of silently making an envelope alias two
    experts.
    """
    granularity = int(envelope_bytes)
    if granularity <= 0:
        raise ValueError("envelope_bytes must be positive")
    owners = existing_owners if existing_owners is not None else {}
    record = ExpertRecord(layer_id, global_expert_id)
    for ptr, size_bytes in slices:
        start = int(ptr)
        end = start + int(size_bytes)
        if start <= 0 or int(size_bytes) <= 0 or end <= start:
            raise ValueError("expert slice geometry must be positive")
        first = (start // granularity) * granularity
        last = ((end + granularity - 1) // granularity) * granularity
        for base in range(first, last, granularity):
            region_id = (
                f"weights:{int(layer_id)}:{int(global_expert_id)}:envelope:"
                f"{base:x}:{granularity:x}"
            )
            geometry_key = f"{base:x}:{granularity:x}"
            owner = owners.get(geometry_key)
            if owner is not None and owner != record.key:
                raise ValueError(
                    f"envelope {region_id} is shared by experts "
                    f"{owner} and {record.key}"
                )
            owners[geometry_key] = record.key
            if region_id not in record.envelope_region_ids:
                record.envelope_region_ids.append(region_id)
    if not record.envelope_region_ids:
        raise ValueError("expert has no envelope ranges")
    return record


class ExplicitRegionRegistry:
    """Small, thread-safe-free registry used at the application boundary.

    The optional executor receives ``(base, length)`` ranges and returns either
    a truthy value or a result object containing ``ok``.  A failed item is
    always recorded as ``UNKNOWN``.
    """

    def __init__(self) -> None:
        self.regions: dict[str, KernelRegion] = {}

    def register_region(
        self,
        region_id: str,
        base: int,
        length: int,
        state: RegionState | str = RegionState.UNKNOWN,
    ) -> KernelRegion:
        candidate = KernelRegion(region_id, base, length, normalize_region_state(state))
        current = self.regions.get(candidate.region_id)
        if current is not None and (current.base, current.length) != (
            candidate.base,
            candidate.length,
        ):
            raise ValueError(f"region geometry changed: {candidate.region_id}")
        if current is not None:
            current.state = candidate.state
            return current
        self.regions[candidate.region_id] = candidate
        return candidate

    @staticmethod
    def _execute(
        region: KernelRegion,
        target: RegionState,
        executor: Callable[[int, int], object] | None,
    ) -> bool:
        if target == RegionState.CPU and region.state == RegionState.CPU:
            return True
        if target == RegionState.GPU and region.state == RegionState.GPU:
            return True
        # UNKNOWN must run a conservative physical operation.  Without an
        # executor there is no evidence that placement succeeded.
        if region.state == RegionState.UNKNOWN and executor is None:
            return False
        region.state = (
            RegionState.PREFETCHING_CPU
            if target == RegionState.CPU
            else RegionState.PREFETCHING_GPU
        )
        try:
            result = True if executor is None else executor(region.base, region.length)
            ok = bool(result.get("ok")) if isinstance(result, Mapping) else bool(result)
        except Exception:
            ok = False
        region.state = target if ok else RegionState.UNKNOWN
        return ok

    def move_regions(
        self,
        region_ids: Iterable[str],
        target: RegionState | str,
        executor: Callable[[int, int], object] | None = None,
    ) -> dict[str, object]:
        destination = normalize_region_state(target)
        if destination not in (RegionState.GPU, RegionState.CPU):
            raise ValueError("target must be GPU or CPU")
        ids = [str(region_id) for region_id in region_ids]
        statuses: dict[str, bool] = {}
        for region_id in ids:
            region = self.regions.get(region_id)
            statuses[region_id] = (
                False
                if region is None
                else self._execute(region, destination, executor)
            )
        return {
            "ok": all(statuses.values()),
            "statuses": statuses,
            "failed_region_ids": [key for key, value in statuses.items() if not value],
        }

    def release_regions(
        self,
        region_ids: Iterable[str],
        executor: Callable[[int, int], object] | None = None,
    ) -> dict[str, object]:
        return self.move_regions(region_ids, RegionState.CPU, executor)

    def restore_regions(
        self,
        region_ids: Iterable[str],
        executor: Callable[[int, int], object] | None = None,
    ) -> dict[str, object]:
        return self.move_regions(region_ids, RegionState.GPU, executor)

    def query_region_states(self, region_ids: Iterable[str]) -> dict[str, RegionState]:
        return {
            str(region_id): self.regions[str(region_id)].state
            for region_id in region_ids
            if str(region_id) in self.regions
        }
