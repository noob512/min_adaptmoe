#!/usr/bin/env python3
"""Snapshot and diagnose the repository's UVM migration profile.

The modified driver exposes cumulative counters in
``/proc/driver/nvidia-uvm/latency_stats``.  This tool takes deltas around an
isolated command, derives non-additive timing ratios, and optionally scans a
range manifest for virtual-layout fragmentation.

It deliberately does not claim that a contiguous managed virtual allocation
has contiguous CPU backing.  ``copy_push_memcopy.bytes / calls`` is reported as
an indirect copy-method/run-size proxy; actual CPU chunk allocation outcomes
are not exported by the current driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")
DEFAULT_RESET = Path("/proc/driver/nvidia-uvm/latency_stats_reset")
DEFAULT_LOADED_SRCVERSION = Path("/sys/module/nvidia_uvm/srcversion")
DEFAULT_MODULE_PARAMETERS = Path("/sys/module/nvidia_uvm/parameters")
DEFAULT_VA_BLOCK_BYTES = 2 * 1024 * 1024

PROFILE_FIELDS = ("calls", "ns", "pages", "bytes")
PROFILE_SECTION = "migrate_profile"
PROFILE_STAGES = (
    "api_total",
    "api_locks_lookup",
    "api_core",
    "api_tracker_wait",
    "core_total",
    "core_first_pass",
    "core_second_pass",
    "ranges_total",
    "range_total",
    "preunmap",
    "multi_block_total",
    "block_create",
    "block_lock_migrate",
    "block_migrate_total",
    "block_make_resident",
    "block_map",
    "make_resident_copy",
    "make_resident_finish",
    "read_dup_total",
    "read_dup_prep",
    "read_dup_populate",
    "read_dup_copy",
    "read_dup_finish",
    "unmap",
    "populate_total",
    "populate_gpu",
    "populate_cpu",
    "copy_resident_total",
    "copy_between_total",
    "copy_push_begin",
    "copy_push_memcopy",
    "copy_push_end",
    "map_total",
    "map_gpu_to",
    "unmap_mask",
    "unmap_gpu",
    "tlb_phys_invalidate",
)

_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*$")
_PROFILE_RE = re.compile(
    r"^\s+([a-z][a-z0-9_]*)\s+"
    r"calls=(\d+)\s+ns=(\d+)\s+avg_ns=(\d+)\s+"
    r"pages=(\d+)\s+bytes=(\d+)\s*$"
)
_COUNTER_RE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s+(0x[0-9a-fA-F]+|\d+)\s*$")


def _integer(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def parse_latency_stats(text: str) -> dict[str, Any]:
    """Parse the procfs format into JSON-ready sections and scalar counters."""
    sections: dict[str, dict[str, dict[str, int]]] = {}
    counters: dict[str, int] = {}
    current_section: str | None = None

    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        section_match = _SECTION_RE.fullmatch(line)
        if section_match:
            current_section = section_match.group(1)
            sections.setdefault(current_section, {})
            continue

        profile_match = _PROFILE_RE.fullmatch(line)
        if profile_match:
            if current_section is None:
                raise ValueError(
                    f"profile row before section at line {line_number}: {line!r}"
                )
            name, calls, ns, _avg_ns, pages, bytes_ = profile_match.groups()
            sections[current_section][name] = {
                "calls": int(calls),
                "ns": int(ns),
                "pages": int(pages),
                "bytes": int(bytes_),
            }
            continue

        counter_match = _COUNTER_RE.fullmatch(line)
        if counter_match:
            name, value = counter_match.groups()
            counters[name] = int(value, 0)
            current_section = None
            continue

        raise ValueError(f"unrecognized latency_stats line {line_number}: {line!r}")

    if PROFILE_SECTION not in sections:
        raise ValueError(f"missing {PROFILE_SECTION!r} section")
    return {"sections": sections, "counters": counters}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_module_parameters() -> dict[str, str]:
    names = (
        "uvm_cpu_chunk_allocation_sizes",
        "uvm_perf_prefetch_enable",
        "uvm_perf_prefetch_threshold",
        "uvm_perf_prefetch_min_faults",
        "uvm_perf_migrate_cpu_preunmap_enable",
        "uvm_perf_migrate_cpu_preunmap_block_order",
    )
    values: dict[str, str] = {}
    for name in names:
        value = _read_text(DEFAULT_MODULE_PARAMETERS / name)
        if value is not None:
            values[name] = value

    raw_mask = values.get("uvm_cpu_chunk_allocation_sizes")
    if raw_mask is not None:
        try:
            mask = int(raw_mask, 0)
            values["uvm_cpu_chunk_allocation_sizes_enabled"] = ",".join(
                str(size)
                for size in (4096, 64 * 1024, 2 * 1024 * 1024)
                if mask & size
            )
        except ValueError:
            pass
    return values


def _parse_buddyinfo(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    zones: list[dict[str, Any]] = []
    totals: list[int] = []
    for line in text.splitlines():
        match = re.fullmatch(r"Node\s+(\d+),\s+zone\s+(\S+)\s+(.+)", line.strip())
        if not match:
            continue
        node, zone, counts_text = match.groups()
        counts = [int(value) for value in counts_text.split()]
        if len(totals) < len(counts):
            totals.extend([0] * (len(counts) - len(totals)))
        for order, count in enumerate(counts):
            totals[order] += count
        zones.append({"node": int(node), "zone": zone, "free_blocks": counts})
    if not zones:
        return None
    page_size = os.sysconf("SC_PAGE_SIZE")
    return {
        "page_size": page_size,
        "zones": zones,
        "total_free_blocks_by_order": totals,
        "total_free_bytes_by_order": [
            count * page_size * (1 << order) for order, count in enumerate(totals)
        ],
        "note": (
            "System-wide buddy availability is contextual only; it does not "
            "identify the physical chunks backing this UVM allocation."
        ),
    }


def module_srcversion(module_path: Path | None) -> str | None:
    if module_path is None:
        return None
    try:
        result = subprocess.run(
            ["modinfo", "-F", "srcversion", str(module_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def take_snapshot(stats_path: Path, module_path: Path | None = None) -> dict[str, Any]:
    try:
        raw = stats_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"cannot read {stats_path}: {error}. The loaded nvidia_uvm module "
            "may not be the instrumented workspace build, or debug procfs may "
            "be disabled."
        ) from error

    return {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "stats_path": str(stats_path),
        "stats_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "loaded_srcversion": _read_text(DEFAULT_LOADED_SRCVERSION),
        "workspace_module": str(module_path) if module_path is not None else None,
        "workspace_module_srcversion": module_srcversion(module_path),
        "module_parameters": _read_module_parameters(),
        "buddyinfo": _parse_buddyinfo(_read_text(Path("/proc/buddyinfo"))),
        "profile": parse_latency_stats(raw),
    }


def _profile_from_document(document: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = document.get("profile", document)
    if not isinstance(profile, Mapping):
        raise ValueError("snapshot does not contain an object profile")
    return profile


def delta_profiles(before_doc: Mapping[str, Any], after_doc: Mapping[str, Any]) -> dict[str, Any]:
    before = _profile_from_document(before_doc)
    after = _profile_from_document(after_doc)
    before_sections = before.get("sections", {})
    after_sections = after.get("sections", {})
    if not isinstance(before_sections, Mapping) or not isinstance(after_sections, Mapping):
        raise ValueError("profile sections must be objects")

    monotonic = True
    negative_paths: list[str] = []
    sections: dict[str, dict[str, dict[str, int]]] = {}
    for section_name in sorted(set(before_sections) | set(after_sections)):
        before_section = before_sections.get(section_name, {})
        after_section = after_sections.get(section_name, {})
        if not isinstance(before_section, Mapping) or not isinstance(after_section, Mapping):
            continue
        output_section: dict[str, dict[str, int]] = {}
        for stage_name in sorted(set(before_section) | set(after_section)):
            before_stage = before_section.get(stage_name, {})
            after_stage = after_section.get(stage_name, {})
            if not isinstance(before_stage, Mapping) or not isinstance(after_stage, Mapping):
                continue
            values: dict[str, int] = {}
            for field in PROFILE_FIELDS:
                value = int(after_stage.get(field, 0)) - int(before_stage.get(field, 0))
                values[field] = value
                if value < 0:
                    monotonic = False
                    negative_paths.append(f"sections.{section_name}.{stage_name}.{field}")
            output_section[stage_name] = values
        sections[section_name] = output_section

    before_counters = before.get("counters", {})
    after_counters = after.get("counters", {})
    if not isinstance(before_counters, Mapping) or not isinstance(after_counters, Mapping):
        raise ValueError("profile counters must be objects")
    counters: dict[str, int] = {}
    for name in sorted(set(before_counters) | set(after_counters)):
        value = int(after_counters.get(name, 0)) - int(before_counters.get(name, 0))
        counters[name] = value
        if value < 0:
            monotonic = False
            negative_paths.append(f"counters.{name}")

    return {
        "sections": sections,
        "counters": counters,
        "monotonic": monotonic,
        "negative_paths": negative_paths,
    }


def _stage(profile: Mapping[str, Any], name: str) -> dict[str, int]:
    sections = profile.get("sections", {})
    if not isinstance(sections, Mapping):
        return {field: 0 for field in PROFILE_FIELDS}
    migrate = sections.get(PROFILE_SECTION, {})
    if not isinstance(migrate, Mapping):
        return {field: 0 for field in PROFILE_FIELDS}
    raw = migrate.get(name, {})
    if not isinstance(raw, Mapping):
        return {field: 0 for field in PROFILE_FIELDS}
    return {field: int(raw.get(field, 0)) for field in PROFILE_FIELDS}


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _time_row(stage: dict[str, int], api_total_ns: int) -> dict[str, Any]:
    return {
        **stage,
        "ms": stage["ns"] / 1_000_000.0,
        "share_of_api_total": _ratio(stage["ns"], api_total_ns),
        "avg_ns_per_call": _ratio(stage["ns"], stage["calls"]),
        "avg_bytes_per_call": _ratio(stage["bytes"], stage["calls"]),
    }


def diagnose_profile(
    profile: Mapping[str, Any],
    *,
    elapsed_ns: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """Derive ratios without summing nested/inclusive profile stages."""
    page = int(page_size or os.sysconf("SC_PAGE_SIZE"))
    stages = {name: _stage(profile, name) for name in PROFILE_STAGES}
    api_total = stages["api_total"]
    api_total_ns = api_total["ns"]

    timing = {
        name: _time_row(stage, api_total_ns) for name, stage in stages.items()
    }
    api_ratios = {
        "locks_lookup": _ratio(stages["api_locks_lookup"]["ns"], api_total_ns),
        "core": _ratio(stages["api_core"]["ns"], api_total_ns),
        "tracker_wait": _ratio(stages["api_tracker_wait"]["ns"], api_total_ns),
    }

    ranked_stage_names = (
        "api_locks_lookup",
        "block_create",
        "block_lock_migrate",
        "block_make_resident",
        "block_map",
        "populate_gpu",
        "populate_cpu",
        "copy_push_begin",
        "copy_push_memcopy",
        "copy_push_end",
        "make_resident_finish",
        "map_gpu_to",
        "unmap_mask",
        "unmap_gpu",
        "tlb_phys_invalidate",
    )
    ranked = sorted(
        (
            {
                "stage": name,
                "ns": stages[name]["ns"],
                "ms": stages[name]["ns"] / 1_000_000.0,
                "share_of_api_total": _ratio(stages[name]["ns"], api_total_ns),
            }
            for name in ranked_stage_names
            if stages[name]["calls"] or stages[name]["ns"]
        ),
        key=lambda row: int(row["ns"]),
        reverse=True,
    )

    copy_method = stages["copy_push_memcopy"]
    method_bytes = _ratio(copy_method["bytes"], copy_method["calls"])
    methods_per_block = _ratio(
        copy_method["calls"], stages["block_migrate_total"]["calls"]
    )
    if method_bytes is None:
        run_proxy = "no_copy_methods_observed"
    elif method_bytes <= 2 * page:
        run_proxy = "page_granular_copy_methods"
    elif method_bytes <= 64 * 1024:
        run_proxy = "small_copy_methods"
    elif method_bytes >= 1024 * 1024:
        run_proxy = "large_copy_methods"
    else:
        run_proxy = "medium_copy_methods"

    core_ratio = api_ratios["core"] or 0.0
    wait_ratio = api_ratios["tracker_wait"] or 0.0
    lookup_ratio = api_ratios["locks_lookup"] or 0.0
    if api_total_ns <= 0:
        dominant = "no_migration_profile_activity"
    elif wait_ratio >= 0.55:
        dominant = "tracker_wait_or_device_work_dominant"
    elif core_ratio >= 0.55:
        dominant = "host_driver_core_dominant"
    elif lookup_ratio >= 0.35:
        dominant = "lookup_lock_dominant"
    else:
        dominant = "mixed"

    bandwidth: dict[str, float | None] = {
        "actual_copy_gib_per_s_over_command_wall": None,
        "requested_gib_per_s_over_command_wall": None,
    }
    if elapsed_ns and elapsed_ns > 0:
        seconds = elapsed_ns / 1_000_000_000.0
        bandwidth = {
            "actual_copy_gib_per_s_over_command_wall": copy_method["bytes"]
            / (1024**3)
            / seconds,
            "requested_gib_per_s_over_command_wall": api_total["bytes"]
            / (1024**3)
            / seconds,
        }

    return {
        "valid": bool(profile.get("monotonic", True)),
        "api_total": _time_row(api_total, api_total_ns),
        "api_time_ratios": api_ratios,
        "dominant_api_component": dominant,
        "stages": timing,
        "ranked_nonadditive_stage_proxies": ranked,
        "granularity": {
            "api_avg_requested_bytes": _ratio(api_total["bytes"], api_total["calls"]),
            "ranges_per_api": _ratio(
                stages["range_total"]["calls"], api_total["calls"]
            ),
            "va_blocks_per_api": _ratio(
                stages["block_migrate_total"]["calls"], api_total["calls"]
            ),
            "avg_bytes_per_migrated_block_call": _ratio(
                stages["block_migrate_total"]["bytes"],
                stages["block_migrate_total"]["calls"],
            ),
            "block_create_fraction": _ratio(
                stages["block_create"]["calls"],
                stages["block_migrate_total"]["calls"],
            ),
            "copy_method_calls_per_block_call": methods_per_block,
            "copy_method_avg_bytes": method_bytes,
            "copy_method_run_proxy": run_proxy,
        },
        "bandwidth": bandwidth,
        "interpretation_limits": {
            "cpu_physical_contiguity_directly_observed": False,
            "cpu_chunk_size_outcomes_exported": False,
            "copy_method_proxy": (
                "copy_push_memcopy bytes/calls observes CE method granularity. "
                "Small methods are consistent with non-contiguous backing, but "
                "can also result from sparse page masks or partial ranges."
            ),
            "managed_arena_caveat": (
                "One contiguous cudaMallocManaged virtual arena does not prove "
                "that its CPU pages are physically contiguous."
            ),
            "timing_caveat": (
                "Profile stages are nested and must not be summed. "
                "copy_push_memcopy.ns measures command construction, while "
                "api_tracker_wait includes completion/backlog wait."
            ),
        },
    }


def _quantile(sorted_values: Sequence[int], fraction: float) -> int | None:
    if not sorted_values:
        return None
    index = round((len(sorted_values) - 1) * fraction)
    return int(sorted_values[index])


def _manifest_records(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, Mapping):
        raw_records = document.get("ranges", document.get("regions", []))
    else:
        raw_records = document
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("range manifest must be a list or contain a ranges/regions list")

    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        if isinstance(raw, Mapping):
            base = raw.get("base", raw.get("ptr", raw.get("address")))
            length = raw.get("length", raw.get("size", raw.get("bytes")))
            allocation = raw.get(
                "allocation_id", raw.get("allocation", raw.get("owner"))
            )
            name = raw.get("name", raw.get("region_id", str(index)))
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
            base, length = raw[0], raw[1]
            allocation = raw[2] if len(raw) >= 3 else None
            name = str(index)
        else:
            records.append({"index": index, "valid": False, "reason": "bad_record"})
            continue
        try:
            base_int = _integer(base)
            length_int = _integer(length)
        except (TypeError, ValueError):
            records.append({"index": index, "valid": False, "reason": "bad_integer"})
            continue
        valid = base_int > 0 and length_int > 0 and base_int + length_int > base_int
        records.append(
            {
                "index": index,
                "name": str(name),
                "base": base_int,
                "length": length_int,
                "end": base_int + length_int,
                "allocation_id": None if allocation is None else str(allocation),
                "valid": valid,
                "reason": "ok" if valid else "invalid_range",
            }
        )
    return records


def _merge_layout_ranges(
    records: Iterable[Mapping[str, Any]], *, same_allocation_adjacent: bool
) -> list[tuple[int, int]]:
    if not same_allocation_adjacent:
        # Overlap proves that two records refer to the same underlying VA;
        # mere adjacency does not prove a shared CUDA allocation.
        ranges = sorted(
            (int(record["base"]), int(record["end"]))
            for record in records
            if record.get("valid")
        )
        if not ranges:
            return []
        merged: list[tuple[int, int]] = []
        start, end = ranges[0]
        for next_start, next_end in ranges[1:]:
            if next_start < end:
                end = max(end, next_end)
            else:
                merged.append((start, end))
                start, end = next_start, next_end
        merged.append((start, end))
        return merged

    groups: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        if not record.get("valid"):
            continue
        allocation = record.get("allocation_id")
        if same_allocation_adjacent and allocation is not None:
            key = f"known:{allocation}"
        else:
            # Unknown allocation boundaries must never be crossed merely
            # because separate cudaMallocManaged VAs happen to touch.
            key = f"record:{record['index']}"
        groups.setdefault(key, []).append((int(record["base"]), int(record["end"])))

    merged: list[tuple[int, int]] = []
    for group in groups.values():
        group.sort()
        start, end = group[0]
        for next_start, next_end in group[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                merged.append((start, end))
                start, end = next_start, next_end
        merged.append((start, end))
    return sorted(merged)


def scan_layout(document: Any, va_block_bytes: int = DEFAULT_VA_BLOCK_BYTES) -> dict[str, Any]:
    records = _manifest_records(document)
    valid = [record for record in records if record.get("valid")]
    valid.sort(key=lambda record: (int(record["base"]), int(record["end"])))
    lengths = sorted(int(record["length"]) for record in valid)

    overlaps = 0
    touching = 0
    gaps: list[int] = []
    previous_end: int | None = None
    for record in valid:
        base = int(record["base"])
        end = int(record["end"])
        if previous_end is not None:
            if base < previous_end:
                overlaps += 1
            elif base == previous_end:
                touching += 1
            else:
                gaps.append(base - previous_end)
        previous_end = max(previous_end or end, end)

    conservative = _merge_layout_ranges(valid, same_allocation_adjacent=False)
    allocation_aware = _merge_layout_ranges(valid, same_allocation_adjacent=True)
    block_intersections: dict[int, int] = {}
    block_intervals: dict[int, list[tuple[int, int]]] = {}
    for record in valid:
        base = int(record["base"])
        end = int(record["end"])
        first = base // va_block_bytes
        last = (end - 1) // va_block_bytes
        for block in range(first, last + 1):
            block_start = block * va_block_bytes
            interval_start = max(base, block_start)
            interval_end = min(end, block_start + va_block_bytes)
            block_intersections[block] = block_intersections.get(block, 0) + 1
            block_intervals.setdefault(block, []).append(
                (interval_start, interval_end)
            )

    block_covered: dict[int, int] = {}
    for block, intervals in block_intervals.items():
        intervals.sort()
        covered = 0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered += end - start
                start, end = next_start, next_end
        block_covered[block] = covered + end - start

    allocation_ids = {
        str(record["allocation_id"])
        for record in valid
        if record.get("allocation_id") is not None
    }
    all_allocation_ids_known = bool(valid) and all(
        record.get("allocation_id") is not None for record in valid
    )
    span = int(valid[-1]["end"]) - int(valid[0]["base"]) if valid else 0
    total_bytes = sum(lengths)
    intersection_values = list(block_intersections.values())
    return {
        "valid_range_count": len(valid),
        "invalid_range_count": len(records) - len(valid),
        "total_range_bytes": total_bytes,
        "virtual_span_bytes": span,
        "virtual_span_fill_fraction": _ratio(total_bytes, span),
        "range_length_bytes": {
            "min": min(lengths) if lengths else None,
            "p50": _quantile(lengths, 0.50),
            "p90": _quantile(lengths, 0.90),
            "max": max(lengths) if lengths else None,
            "mean": statistics.fmean(lengths) if lengths else None,
        },
        "adjacency": {
            "overlapping_neighbor_pairs": overlaps,
            "touching_neighbor_pairs": touching,
            "gap_count": len(gaps),
            "gap_bytes_total": sum(gaps),
            "gap_bytes_max": max(gaps) if gaps else 0,
        },
        "coalescing": {
            "conservative_call_count": len(conservative),
            "allocation_aware_call_count": len(allocation_aware),
            "known_allocation_count": len(allocation_ids),
            "all_allocation_ids_known": all_allocation_ids_known,
            "note": (
                "Touching ranges are coalesced only when they share an explicit "
                "allocation_id; CUDA prefetch cannot safely cross an unknown "
                "cudaMallocManaged allocation boundary."
            ),
        },
        "va_blocks": {
            "block_bytes": va_block_bytes,
            "touched": len(block_intersections),
            "fully_covered": sum(
                1 for covered in block_covered.values() if covered >= va_block_bytes
            ),
            "partially_covered": sum(
                1 for covered in block_covered.values() if covered < va_block_bytes
            ),
            "range_intersections": sum(intersection_values),
            "mean_range_intersections_per_block": (
                statistics.fmean(intersection_values) if intersection_values else None
            ),
            "max_range_intersections_per_block": (
                max(intersection_values) if intersection_values else 0
            ),
        },
        "physical_contiguity_directly_observed": False,
        "managed_arena_caveat": (
            "Virtual adjacency and allocation identity do not prove contiguous "
            "CPU physical backing. Use isolated granularity sweeps plus the "
            "copy-method proxy, or add CPU chunk allocation-result counters."
        ),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(document: Any, output: Path | None) -> None:
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output is None or str(output) == "-":
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def _validate_module(snapshot: Mapping[str, Any], allow_mismatch: bool) -> None:
    loaded = snapshot.get("loaded_srcversion")
    workspace = snapshot.get("workspace_module_srcversion")
    if loaded and workspace and loaded != workspace and not allow_mismatch:
        raise RuntimeError(
            "loaded/workspace nvidia_uvm srcversion mismatch: "
            f"loaded={loaded} workspace={workspace}; refusing to profile the "
            "wrong driver (pass --allow-module-mismatch only for deliberate use)"
        )


def _run_command(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("run requires a command after --")

    if args.reset:
        try:
            args.reset_file.write_text("1\n", encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"cannot reset {args.reset_file}: {error}") from error

    before = take_snapshot(args.stats, args.module)
    _validate_module(before, args.allow_module_mismatch)
    start_ns = time.monotonic_ns()
    completed = subprocess.run(command, check=False)
    elapsed_ns = time.monotonic_ns() - start_ns
    after = take_snapshot(args.stats, args.module)
    delta = delta_profiles(before, after)
    layout = scan_layout(_load_json(args.ranges)) if args.ranges else None
    result = {
        "schema_version": 1,
        "label": args.label,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_ns": elapsed_ns,
        "before": before,
        "after": after,
        "delta": delta,
        "diagnosis": diagnose_profile(delta, elapsed_ns=elapsed_ns),
        "layout": layout,
        "global_counter_caveat": (
            "latency_stats is module-global. Run without unrelated UVM work, "
            "and use repeated allocation/range-granularity cases for causality."
        ),
    }
    _write_json(result, args.output)
    return int(completed.returncode)


def _self_test() -> None:
    sample_before = """\
migrate_profile:
  api_total                calls=1 ns=100 avg_ns=100 pages=16 bytes=65536
  api_locks_lookup         calls=1 ns=10 avg_ns=10 pages=16 bytes=65536
  api_core                 calls=1 ns=60 avg_ns=60 pages=16 bytes=65536
  api_tracker_wait         calls=1 ns=30 avg_ns=30 pages=16 bytes=65536
  block_migrate_total      calls=1 ns=50 avg_ns=50 pages=16 bytes=65536
  copy_push_memcopy        calls=16 ns=16 avg_ns=1 pages=16 bytes=65536
expert_batch_profile:
  total                    calls=0 ns=0 avg_ns=0 pages=0 bytes=0
expert_batch_attempts           0
"""
    sample_after = sample_before.replace("calls=1 ns=100", "calls=2 ns=200", 1)
    before = parse_latency_stats(sample_before)
    after = parse_latency_stats(sample_after)
    delta = delta_profiles(before, after)
    assert _stage(delta, "api_total")["calls"] == 1
    assert _stage(delta, "api_total")["ns"] == 100
    diagnosis = diagnose_profile(before, elapsed_ns=1_000_000)
    assert diagnosis["granularity"]["copy_method_avg_bytes"] == 4096.0
    layout = scan_layout(
        {
            "ranges": [
                {"base": "0x200000", "length": 65536, "allocation_id": "a"},
                {"base": "0x210000", "length": 65536, "allocation_id": "a"},
                {"base": "0x300000", "length": 4096, "allocation_id": "b"},
            ]
        }
    )
    assert layout["coalescing"]["allocation_aware_call_count"] == 2
    assert layout["coalescing"]["conservative_call_count"] == 3
    print("self-test: ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="capture current counters")
    snapshot_parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    snapshot_parser.add_argument("--module", type=Path)
    snapshot_parser.add_argument("--output", type=Path)

    analyze_parser = subparsers.add_parser("analyze", help="diff two snapshots")
    analyze_parser.add_argument("before", type=Path)
    analyze_parser.add_argument("after", type=Path)
    analyze_parser.add_argument("--elapsed-ms", type=float)
    analyze_parser.add_argument("--ranges", type=Path)
    analyze_parser.add_argument("--output", type=Path)

    layout_parser = subparsers.add_parser("layout", help="scan a range manifest")
    layout_parser.add_argument("manifest", type=Path)
    layout_parser.add_argument("--va-block-bytes", type=int, default=DEFAULT_VA_BLOCK_BYTES)
    layout_parser.add_argument("--output", type=Path)

    run_parser = subparsers.add_parser("run", help="profile one isolated command")
    run_parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    run_parser.add_argument("--reset-file", type=Path, default=DEFAULT_RESET)
    run_parser.add_argument("--module", type=Path)
    run_parser.add_argument("--allow-module-mismatch", action="store_true")
    run_parser.add_argument("--reset", action="store_true")
    run_parser.add_argument("--ranges", type=Path)
    run_parser.add_argument("--label")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)

    subparsers.add_parser("self-test", help="exercise parser and derived metrics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "snapshot":
        _write_json(take_snapshot(args.stats, args.module), args.output)
        return 0
    if args.subcommand == "analyze":
        before = _load_json(args.before)
        after = _load_json(args.after)
        delta = delta_profiles(before, after)
        elapsed_ns = (
            int(args.elapsed_ms * 1_000_000) if args.elapsed_ms is not None else None
        )
        _write_json(
            {
                "delta": delta,
                "diagnosis": diagnose_profile(delta, elapsed_ns=elapsed_ns),
                "layout": scan_layout(_load_json(args.ranges)) if args.ranges else None,
            },
            args.output,
        )
        return 0
    if args.subcommand == "layout":
        _write_json(
            scan_layout(_load_json(args.manifest), args.va_block_bytes), args.output
        )
        return 0
    if args.subcommand == "run":
        return _run_command(args)
    if args.subcommand == "self-test":
        _self_test()
        return 0
    raise AssertionError(args.subcommand)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
