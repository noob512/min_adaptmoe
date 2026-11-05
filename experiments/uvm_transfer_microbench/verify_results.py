#!/usr/bin/env python3
"""Verify the saved transfer benchmark and UVM driver-profile evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LIVE_SRCVERSION = Path("/sys/module/nvidia_uvm/srcversion")
LIVE_CHUNK_MASK = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_cpu_chunk_allocation_sizes"
)


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in line.split()[1:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        values[key] = value
    return values


def load_text_records(
    path: Path,
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str] | None]:
    results: list[dict[str, str]] = []
    final: dict[str, str] | None = None
    diagnosis: dict[str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("RESULT "):
            results.append(parse_key_values(line))
        elif line.startswith("DIAGNOSIS "):
            diagnosis = parse_key_values(line)
        elif line.startswith("benchmark_result "):
            final = parse_key_values(line)
    if final is None:
        raise ValueError(f"missing benchmark_result in {path}")
    return results, final, diagnosis


def find_result(
    records: list[dict[str, str]], layout: str, backend: str, mode: str
) -> dict[str, str]:
    for record in records:
        if (
            record.get("layout") == layout
            and record.get("backend") == backend
            and record.get("mode") == mode
        ):
            return record
    raise ValueError(f"missing RESULT {layout}/{backend}/{mode}")


def number(record: dict[str, str], name: str) -> float:
    value = float(record[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}={record[name]}")
    return value


def load_profile(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("returncode") != 0:
        raise ValueError(f"profile command failed: {path}")
    if not document.get("delta", {}).get("monotonic", False):
        raise ValueError(f"non-monotonic driver profile: {path}")
    return document


def profile_value(document: dict[str, Any], stage: str, field: str) -> int:
    return int(document["delta"]["sections"]["migrate_profile"][stage][field])


def profile_srcversion(document: dict[str, Any]) -> str:
    versions = {
        document[point].get(field)
        for point in ("before", "after")
        for field in ("loaded_srcversion", "workspace_module_srcversion")
    }
    if None in versions or len(versions) != 1:
        raise ValueError("loaded/workspace UVM srcversion mismatch in profile")
    return str(versions.pop())


def profile_chunk_mask(document: dict[str, Any]) -> int:
    masks = {
        int(
            document[point]["module_parameters"][
                "uvm_cpu_chunk_allocation_sizes"
            ],
            0,
        )
        for point in ("before", "after")
    }
    if len(masks) != 1:
        raise ValueError("UVM CPU chunk mask changed while profile was running")
    return masks.pop()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
    print(f"PASS {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", type=Path, default=SCRIPT_DIR / "result_20260814_default_chunks.txt"
    )
    parser.add_argument(
        "--split-profile",
        type=Path,
        default=SCRIPT_DIR / "driver_profile_contiguous_tensor_batch_20260814.json",
    )
    parser.add_argument(
        "--coalesced-profile",
        type=Path,
        default=SCRIPT_DIR / "driver_profile_contiguous_coalesced_20260814.json",
    )
    parser.add_argument(
        "--coalesced-result",
        type=Path,
        default=SCRIPT_DIR / "result_contiguous_coalesced_20260814.txt",
    )
    parser.add_argument(
        "--four-k-result",
        type=Path,
        default=SCRIPT_DIR / "result_contiguous_coalesced_4k_chunks_20260814.txt",
    )
    parser.add_argument(
        "--four-k-profile",
        type=Path,
        default=SCRIPT_DIR
        / "driver_profile_contiguous_coalesced_4k_chunks_20260814.json",
    )
    parser.add_argument("--equivalent-ratio", type=float, default=0.90)
    parser.add_argument("--minimum-coalesce-speedup", type=float, default=1.50)
    parser.add_argument("--minimum-4k-slowdown", type=float, default=2.00)
    parser.add_argument(
        "--check-live-state",
        action="store_true",
        help="also require the loaded module to match the saved default state",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records, final, diagnosis = load_text_records(args.full)
    require(final.get("functional_success") == "1", "full functional_success=1")
    require(len(records) == 17, "full matrix contains 17 cases")
    require(final.get("total_cases") == "17", "full result declares total_cases=17")
    require(
        final.get("cpu_virtual_layout_controlled") == "1"
        and final.get("cpu_physical_contiguity_controlled") == "0",
        "result distinguishes controlled VA layout from uncontrolled PFN continuity",
    )
    require(
        all(record.get("functional_ok") == "1" for record in records),
        "all full-matrix checksums and placements pass",
    )

    managed = [record for record in records if record.get("backend") == "managed_prefetch"]
    require(
        all(
            record.get("in_place_va") == "1"
            and record.get("same_consumer_va") == "1"
            for record in managed
        ),
        "all managed-prefetch cases preserve their consumer VA",
    )

    pinned = find_result(records, "multi_alloc", "pinned_copy", "tensor_batch")
    multi = find_result(records, "multi_alloc", "managed_prefetch", "tensor_batch")
    interleaved = find_result(
        records, "arena_interleaved", "managed_prefetch", "tensor_batch"
    )
    split = find_result(
        records, "arena_contiguous", "managed_prefetch", "tensor_batch"
    )
    coalesced = find_result(
        records, "arena_contiguous", "managed_prefetch", "coalesced_batch"
    )
    uva = find_result(records, "arena_contiguous", "uva_direct", "direct")

    pinned_bw = number(pinned, "migration_gib_s")
    multi_bw = number(multi, "migration_gib_s")
    interleaved_bw = number(interleaved, "migration_gib_s")
    split_bw = number(split, "migration_gib_s")
    coalesced_bw = number(coalesced, "migration_gib_s")
    uva_bw = number(uva, "effective_gib_s")
    parity = coalesced_bw / pinned_bw
    coalesce_speedup = coalesced_bw / split_bw
    fragmented_spread = min(multi_bw, interleaved_bw, split_bw) / max(
        multi_bw, interleaved_bw, split_bw
    )

    require(
        parity >= args.equivalent_ratio,
        f"coalesced UVM reaches {parity:.3f}x pinned copy",
    )
    require(
        coalesce_speedup >= args.minimum_coalesce_speedup,
        f"128-to-1 range coalescing gives {coalesce_speedup:.3f}x",
    )
    require(
        fragmented_spread >= 0.80,
        f"same-range-count layouts stay within 20% ({fragmented_spread:.3f})",
    )
    require(uva.get("functional_ok") == "1", f"UVA direct read passes at {uva_bw:.3f} GiB/s")
    require(diagnosis is not None, "full result contains DIAGNOSIS")
    require(
        diagnosis.get("conclusion")
        in {
            "per_range_uvm_driver_or_migration_metadata_likely",
            "per_range_uvm_driver_orchestration_likely",
        },
        "DIAGNOSIS records the expected per-range orchestration heuristic",
    )
    require(
        math.isclose(
            float(diagnosis["uvm_coalesce_speedup"]),
            coalesce_speedup,
            rel_tol=5e-4,
        ),
        "DIAGNOSIS coalescing speedup matches RESULT records",
    )

    split_profile = load_profile(args.split_profile)
    coalesced_profile = load_profile(args.coalesced_profile)
    split_calls = profile_value(split_profile, "api_total", "calls")
    coalesced_calls = profile_value(coalesced_profile, "api_total", "calls")
    split_method = float(
        split_profile["diagnosis"]["granularity"]["copy_method_avg_bytes"]
    )
    coalesced_method = float(
        coalesced_profile["diagnosis"]["granularity"]["copy_method_avg_bytes"]
    )
    split_methods = profile_value(split_profile, "copy_push_memcopy", "calls")
    coalesced_methods = profile_value(
        coalesced_profile, "copy_push_memcopy", "calls"
    )
    split_mask = profile_chunk_mask(split_profile)
    coalesced_mask = profile_chunk_mask(coalesced_profile)
    split_srcversion = profile_srcversion(split_profile)
    coalesced_srcversion = profile_srcversion(coalesced_profile)
    split_api_bytes = profile_value(split_profile, "api_total", "bytes")
    coalesced_api_bytes = profile_value(coalesced_profile, "api_total", "bytes")
    split_copy_bytes = profile_value(split_profile, "copy_push_memcopy", "bytes")
    coalesced_copy_bytes = profile_value(
        coalesced_profile, "copy_push_memcopy", "bytes"
    )
    split_copy_pages = profile_value(split_profile, "copy_push_memcopy", "pages")
    coalesced_copy_pages = profile_value(
        coalesced_profile, "copy_push_memcopy", "pages"
    )
    split_block_calls = profile_value(
        split_profile, "block_migrate_total", "calls"
    )
    coalesced_block_calls = profile_value(
        coalesced_profile, "block_migrate_total", "calls"
    )
    require(split_calls > coalesced_calls * 10, f"range split raises API calls {coalesced_calls}->{split_calls}")
    require(
        split_mask == 2166784 and coalesced_mask == 2166784,
        "default profiles record the 2MiB+64KiB+4KiB chunk mask",
    )
    require(
        split_srcversion == coalesced_srcversion,
        f"default profiles use loaded/workspace UVM srcversion {split_srcversion}",
    )
    require(
        split_method == 2 * 1024**2 and coalesced_method == 2 * 1024**2,
        "default split and coalesced paths both use exact 2MiB copy methods",
    )
    require(
        split_methods == coalesced_methods,
        f"default CE copy descriptor count is unchanged ({split_methods} calls)",
    )
    require(
        split_api_bytes == coalesced_api_bytes
        and split_copy_bytes == coalesced_copy_bytes == 6 * 1024**3
        and split_copy_pages == coalesced_copy_pages
        and split_block_calls == coalesced_block_calls,
        "default profiles have paired API bytes, CE bytes/pages, and VA-block calls",
    )

    default_records, default_final, _ = load_text_records(args.coalesced_result)
    require(
        default_final.get("functional_success") == "1",
        "matched default-chunk coalesced run functional_success=1",
    )
    default_coalesced = find_result(
        default_records, "arena_contiguous", "managed_prefetch", "coalesced_batch"
    )
    four_k_records, four_k_final, _ = load_text_records(args.four_k_result)
    require(four_k_final.get("functional_success") == "1", "forced-4KiB functional_success=1")
    four_k = find_result(
        four_k_records, "arena_contiguous", "managed_prefetch", "coalesced_batch"
    )
    require(
        all(
            default_coalesced.get(field) == four_k.get(field)
            for field in (
                "layout",
                "backend",
                "mode",
                "bytes",
                "logical_ranges",
                "submitted_ranges",
            )
        ),
        "default and forced-4KiB runs use the same transfer configuration",
    )
    default_coalesced_bw = number(default_coalesced, "migration_gib_s")
    four_k_bw = number(four_k, "migration_gib_s")
    physical_slowdown = default_coalesced_bw / four_k_bw
    four_k_profile = load_profile(args.four_k_profile)
    four_k_mask = profile_chunk_mask(four_k_profile)
    four_k_srcversion = profile_srcversion(four_k_profile)
    four_k_method = float(
        four_k_profile["diagnosis"]["granularity"]["copy_method_avg_bytes"]
    )
    require(four_k_mask == 4096, "forced profile records CPU chunk mask=4096")
    require(
        four_k_srcversion == coalesced_srcversion,
        "forced and default profiles use the same loaded/workspace UVM build",
    )
    require(four_k_method == 4096, "forced profile observes exact 4KiB copy methods")
    require(
        profile_value(four_k_profile, "copy_push_memcopy", "bytes")
        == coalesced_copy_bytes
        and profile_value(four_k_profile, "copy_push_memcopy", "pages")
        == coalesced_copy_pages,
        "forced and default profiles move the same CE bytes/pages",
    )
    require(
        physical_slowdown >= args.minimum_4k_slowdown,
        f"forced 4KiB backing is {physical_slowdown:.3f}x slower",
    )

    if args.check_live_state:
        live_srcversion = LIVE_SRCVERSION.read_text(encoding="utf-8").strip()
        live_mask = int(LIVE_CHUNK_MASK.read_text(encoding="utf-8").strip(), 0)
        require(
            live_srcversion == coalesced_srcversion,
            "live loaded UVM srcversion matches the profiled workspace build",
        )
        require(
            live_mask == 2166784,
            "live UVM CPU chunk mask is restored to the default value",
        )

    print(
        "verification_result success=1 "
        f"pinned_gib_s={pinned_bw:.3f} multi_uvm_gib_s={multi_bw:.3f} "
        f"coalesced_uvm_gib_s={coalesced_bw:.3f} uva_gib_s={uva_bw:.3f} "
        f"coalesced_vs_pinned={parity:.3f} "
        f"coalesce_speedup={coalesce_speedup:.3f} "
        f"forced_4k_slowdown={physical_slowdown:.3f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"verification_result success=0 error={error}")
        raise SystemExit(1) from error
