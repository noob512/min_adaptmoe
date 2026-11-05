#!/usr/bin/env python3
"""Summarize CUDA work by E11 scheduler step from an Nsight SQLite export."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start > end:
            merged += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return merged + end - start


def short_operation(name: str) -> str:
    if name.startswith("memcpy:"):
        return name
    if "fused_moe_kernel" in name:
        return "kernel:fused_moe"
    if "cutlass::Kernel" in name:
        return "kernel:cutlass"
    if "flash_fwd_kernel" in name:
        return "kernel:flash_attention"
    if "rms_norm" in name:
        return "kernel:rms_norm"
    return "kernel:" + name[:96]


def largest_gaps(
    operations: list[tuple[int, int, str]], limit: int = 8
) -> list[dict[str, Any]]:
    if not operations:
        return []
    ordered = sorted(operations)
    cursor_end = ordered[0][1]
    cursor_name = ordered[0][2]
    gaps: list[dict[str, Any]] = []
    for start, end, name in ordered[1:]:
        if start > cursor_end:
            gaps.append(
                {
                    "duration_ms": (start - cursor_end) / 1e6,
                    "after": short_operation(cursor_name),
                    "before": short_operation(name),
                }
            )
        if end > cursor_end:
            cursor_end = end
            cursor_name = name
    return sorted(gaps, key=lambda item: item["duration_ms"], reverse=True)[:limit]


def nvtx_ranges(db: sqlite3.Connection) -> dict[int, tuple[int, int, int]]:
    rows = db.execute(
        """
        SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON s.id = n.textId
        WHERE COALESCE(n.text, s.value) LIKE 'e11_engine_step_%'
        """
    )
    result: dict[int, tuple[int, int, int]] = {}
    for start, end, global_tid, name in rows:
        step = int(str(name).rsplit("_", 1)[1])
        result[step] = (int(start), int(end), int(global_tid))
    return result


def summarize_arm(arm_dir: Path) -> dict[str, Any]:
    result = json.loads((arm_dir / "result.json").read_text(encoding="utf-8"))
    db = sqlite3.connect(arm_dir / "timeline.sqlite")
    ranges = nvtx_ranges(db)
    output: list[dict[str, Any]] = []
    for schedule in result["scheduler_steps"]:
        step = int(schedule["scheduler_step"])
        range_start, range_end, range_tid = ranges[step]
        memcpy_rows = db.execute(
            """
            SELECT DISTINCT m.start, m.end, m.bytes, m.copyKind
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
              ON r.correlationId = m.correlationId
            WHERE r.start >= ? AND r.start <= ?
            """,
            (range_start, range_end),
        ).fetchall()
        kernel_rows = db.execute(
            """
            SELECT DISTINCT k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
              ON r.correlationId = k.correlationId
            JOIN StringIds AS s ON s.id = k.demangledName
            WHERE r.start >= ? AND r.start <= ?
            """,
            (range_start, range_end),
        ).fetchall()
        runtime_rows = db.execute(
            """
            SELECT r.start, r.end, s.value
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON s.id = r.nameId
            WHERE r.start >= ? AND r.start <= ?
            """,
            (range_start, range_end),
        ).fetchall()
        osrt_rows = db.execute(
            """
            SELECT o.start, o.end, s.value
            FROM OSRT_API AS o
            JOIN StringIds AS s ON s.id = o.nameId
            WHERE o.globalTid = ? AND o.start >= ? AND o.start <= ?
            """,
            (range_tid, range_start, range_end),
        ).fetchall()
        h2d = [row for row in memcpy_rows if int(row[3]) == 1]
        kernels = [(int(row[0]), int(row[1]), str(row[2])) for row in kernel_rows]
        operations_with_names = [
            (
                int(row[0]),
                int(row[1]),
                "memcpy:" + ("H2D" if int(row[3]) == 1 else str(row[3])),
            )
            for row in memcpy_rows
        ] + [(row[0], row[1], row[2]) for row in kernels]
        operations = [(start, end) for start, end, _ in operations_with_names]
        span_ns = (
            max(end for _, end in operations) - min(start for start, _ in operations)
            if operations
            else 0
        )
        busy_ns = union_ns(operations)
        output.append(
            {
                "step": step,
                "tokens": int(schedule["total_num_scheduled_tokens"]),
                "engine_wall_ms": float(schedule["engine_step_duration_ms"]),
                "nvtx_cpu_ms": (range_end - range_start) / 1e6,
                "gpu_span_ms": span_ns / 1e6,
                "gpu_busy_union_ms": busy_ns / 1e6,
                "gpu_gap_ms": (span_ns - busy_ns) / 1e6,
                "h2d_ms": sum(int(row[1]) - int(row[0]) for row in h2d) / 1e6,
                "h2d_gb": sum(int(row[2]) for row in h2d) / 1e9,
                "h2d_count": len(h2d),
                "kernel_sum_ms": sum(end - start for start, end, _ in kernels)
                / 1e6,
                "fused_moe_kernel_ms": sum(
                    end - start
                    for start, end, name in kernels
                    if "fused_moe_kernel" in name
                )
                / 1e6,
                "kernel_count": len(kernels),
                "cuda_api_sum_ms": sum(int(end) - int(start) for start, end, _ in runtime_rows)
                / 1e6,
                "cuda_api_count": len(runtime_rows),
                "main_thread_osrt_ms": sum(
                    int(end) - int(start) for start, end, _ in osrt_rows
                )
                / 1e6,
                "main_thread_ioctl_ms": sum(
                    int(end) - int(start)
                    for start, end, name in osrt_rows
                    if name == "ioctl"
                )
                / 1e6,
                "main_thread_ioctl_count": sum(
                    name == "ioctl" for _, _, name in osrt_rows
                ),
                "largest_gpu_gaps": largest_gaps(operations_with_names),
            }
        )
    db.close()
    return {"arm": arm_dir.name, "mode": result["mode"], "steps": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    arm_dirs = sorted(args.run_root.glob("arms/*adaptive")) + sorted(
        args.run_root.glob("arms/*native_copy")
    )
    if not arm_dirs:
        raise SystemExit("no Adaptive/native-copy arms found")
    payload = {"run_root": str(args.run_root.resolve()), "arms": []}
    for arm_dir in arm_dirs:
        if not (arm_dir / "timeline.sqlite").is_file():
            raise SystemExit(f"missing timeline.sqlite: {arm_dir}")
        payload["arms"].append(summarize_arm(arm_dir))
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
