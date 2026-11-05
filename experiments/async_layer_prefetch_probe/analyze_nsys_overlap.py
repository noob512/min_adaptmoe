#!/usr/bin/env python3
"""Extract spin-kernel/Unified-HtoD overlap from an Nsight SQLite export."""

import argparse
import sqlite3


def merged_intersection_ns(kernel_start, kernel_end, copies):
    intervals = sorted(
        (max(kernel_start, start), min(kernel_end, end))
        for start, end, _, _ in copies
        if start < kernel_end and end > kernel_start
    )
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", help="SQLite file produced by `nsys export`")
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    if args.layers < 2:
        parser.error("--layers must be at least 2")

    connection = sqlite3.connect(args.sqlite)
    spins = list(
        connection.execute(
            """
            SELECT kernel.start, kernel.end, kernel.streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            JOIN StringIds AS name ON name.id = kernel.shortName
            WHERE name.value = 'spin_kernel'
            ORDER BY kernel.start
            """
        )
    )
    copies = list(
        connection.execute(
            """
            SELECT copy.start, copy.end, copy.bytes, copy.streamId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copy
            JOIN ENUM_CUDA_MEMCPY_OPER AS kind ON kind.id = copy.copyKind
            JOIN ENUM_CUDA_UNIF_MEM_MIGRATION AS cause
              ON cause.id = copy.migrationCause
            WHERE kind.name = 'CUDA_MEMCPY_KIND_UVM_HTOD'
              AND cause.name = 'CUDA_UNIFIED_MEMORY_MIGRATION_CAUSE_USER'
            ORDER BY copy.start
            """
        )
    )
    expected_spins = 2 * args.layers
    if len(spins) != expected_spins:
        raise SystemExit(
            f"expected {expected_spins} spin kernels from repeats=1, got {len(spins)}"
        )

    first_timestamp = connection.execute(
        """
        SELECT MIN(timestamp) FROM (
            SELECT MIN(start) AS timestamp FROM CUPTI_ACTIVITY_KIND_KERNEL
            UNION ALL
            SELECT MIN(start) AS timestamp FROM CUPTI_ACTIVITY_KIND_MEMCPY
        )
        """
    ).fetchone()[0]
    print(
        f"spin_count={len(spins)} uvm_htod_count={len(copies)} "
        f"uvm_htod_bytes={sum(copy[2] for copy in copies)}"
    )
    print("idx mode start_ms end_ms duration_ms copy_ops copy_bytes intersection_ms")
    for index, (start, end, _) in enumerate(spins):
        mode = "sequential" if index < args.layers else "async"
        hits = [copy for copy in copies if copy[0] < end and copy[1] > start]
        print(
            index,
            mode,
            f"{(start - first_timestamp) / 1e6:.6f}",
            f"{(end - first_timestamp) / 1e6:.6f}",
            f"{(end - start) / 1e6:.6f}",
            len(hits),
            sum(copy[2] for copy in hits),
            f"{merged_intersection_ns(start, end, hits) / 1e6:.6f}",
        )

    for mode, kernels in (
        ("sequential", spins[: args.layers]),
        ("async", spins[args.layers :]),
    ):
        hit_indices = set()
        for copy_index, copy in enumerate(copies):
            for start, end, _ in kernels:
                if copy[0] < end and copy[1] > start:
                    hit_indices.add(copy_index)
        intersection_ns = sum(
            merged_intersection_ns(start, end, copies)
            for start, end, _ in kernels
        )
        print(
            f"aggregate mode={mode} hit_ops={len(hit_indices)} "
            f"hit_bytes={sum(copies[index][2] for index in hit_indices)} "
            f"intersection_ms={intersection_ns / 1e6:.6f}"
        )


if __name__ == "__main__":
    main()
