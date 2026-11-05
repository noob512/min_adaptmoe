#!/usr/bin/env python3
"""Verify and aggregate a formal E4 C1--C32 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PCIE_GEN5_X16_BYTES_S = 32e9 * (128.0 / 130.0) * 16.0 / 8.0
LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def median(values: list[float]) -> float:
    require(bool(values), "median of empty sample")
    return statistics.median(values)


def bootstrap_median_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    require(bool(values), "bootstrap of empty sample")
    if len(values) == 1:
        return values[0], values[0]
    samples = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(10_000)
    )
    return samples[249], samples[9749]


def read_nvml(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            epoch = number(row.get("epoch_seconds"))
            if epoch is None:
                continue
            record = {"epoch_seconds": epoch}
            for key in (
                "pcie_rx_kib_s",
                "pcie_tx_kib_s",
                "gpu_util_pct",
                "memory_util_pct",
                "power_w",
                "sm_clock_mhz",
                "pcie_gen",
                "pcie_width",
            ):
                value = number(row.get(key))
                if value is not None:
                    record[key] = value
            records.append(record)
    return records


def window_rows(records: list[dict[str, float]], start: float, end: float) -> list[dict[str, float]]:
    selected = [row for row in records if start <= row["epoch_seconds"] <= end]
    if not selected:
        midpoint = (start + end) / 2.0
        selected = sorted(records, key=lambda row: abs(row["epoch_seconds"] - midpoint))[:1]
    return selected


def mean_field(records: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in records if key in row]
    return statistics.fmean(values) if values else None


def max_field(records: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in records if key in row]
    return max(values) if values else None


def read_gpm(path: Path) -> list[dict[str, float]]:
    names = [
        "gpu", "power_w", "gtemp", "mtemp", "sm_legacy_pct", "mem_legacy_pct",
        "enc", "dec", "jpg", "ofa", "mclk", "pclk", "gract_pct",
        "sm_activity_pct", "sm_occupancy_pct", "gpu_dram_activity_pct",
        "pcie_tx_mib_s", "pcie_rx_mib_s",
    ]
    records: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 20 or not fields[0].isdigit() or len(fields[0]) != 8:
            continue
        timestamp = datetime.strptime(fields[0] + " " + fields[1], "%Y%m%d %H:%M:%S")
        record: dict[str, float] = {
            "epoch_seconds": timestamp.replace(tzinfo=LOCAL_TZ).timestamp()
        }
        for key, raw in zip(names, fields[2:]):
            value = number(raw)
            if value is not None:
                record[key] = value
        records.append(record)
    return records


def read_sar(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    date_value: datetime | None = None
    for line in lines:
        if line.startswith("Linux "):
            for token in line.split():
                try:
                    date_value = datetime.strptime(token, "%m/%d/%y")
                    break
                except ValueError:
                    continue
            continue
        fields = line.split()
        if date_value is None or len(fields) < 12 or fields[1] != "all":
            continue
        try:
            clock = datetime.strptime(fields[0], "%H:%M:%S")
        except ValueError:
            continue
        idle = number(fields[-1])
        if idle is None:
            continue
        timestamp = date_value.replace(
            hour=clock.hour, minute=clock.minute, second=clock.second, tzinfo=LOCAL_TZ
        )
        records.append({"epoch_seconds": timestamp.timestamp(), "cpu_busy_pct": 100.0 - idle})
    return records


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(record.get(key)) for key in fields})


def svg_plot(path: Path, aggregate: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    width, height = 1100, 760
    left, right = 90, 1040
    panels = [(70, 280), (300, 510), (530, 720)]
    xs = [left + (row["concurrency"] - 1) / 31.0 * (right - left) for row in aggregate]
    series = [
        ("TPOT (ms)", "request_tpot_median_ms", "#b91c1c"),
        ("PCIe RX (GiB/s)", "pcie_rx_gib_s", "#1d4ed8"),
        ("GPM SM activity / occupancy (%)", "gpm_sm_activity_pct", "#047857"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">E4 remote-read concurrency scan</text>',
    ]
    for panel_index, ((top, bottom), (label, key, color)) in enumerate(zip(panels, series)):
        values = [float(row[key]) for row in aggregate if row.get(key) is not None]
        maximum = max(values) * 1.12 if values else 1.0
        minimum = 0.0
        parts.extend([
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#777"/>',
            f'<text x="18" y="{(top + bottom) / 2}" transform="rotate(-90 18 {(top + bottom) / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">{label}</text>',
            f'<text x="{left - 12}" y="{top + 5}" text-anchor="end" font-family="monospace" font-size="12">{maximum:.1f}</text>',
            f'<text x="{left - 12}" y="{bottom + 5}" text-anchor="end" font-family="monospace" font-size="12">0</text>',
        ])
        points: list[str] = []
        for x, row in zip(xs, aggregate):
            value = row.get(key)
            if value is None:
                continue
            y = bottom - (float(value) - minimum) / (maximum - minimum) * (bottom - top)
            points.append(f"{x:.2f},{y:.2f}")
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.7" fill="{color}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        if panel_index == 2:
            occ_points: list[str] = []
            for x, row in zip(xs, aggregate):
                value = row.get("gpm_sm_occupancy_pct")
                if value is None:
                    continue
                y = bottom - float(value) / maximum * (bottom - top)
                occ_points.append(f"{x:.2f},{y:.2f}")
            parts.append(f'<polyline points="{" ".join(occ_points)}" fill="none" stroke="#7c3aed" stroke-width="2" stroke-dasharray="6 4"/>')
            parts.append(f'<text x="{right - 180}" y="{top + 18}" font-family="sans-serif" font-size="12" fill="#7c3aed">dashed: SM occupancy</text>')
        point = decision.get("first_sustained_degradation_concurrency")
        if point is not None:
            x = left + (int(point) - 1) / 31.0 * (right - left)
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#111" stroke-dasharray="3 4"/>')
    for concurrency in (1, 4, 8, 12, 16, 20, 24, 28, 32):
        x = left + (concurrency - 1) / 31.0 * (right - left)
        parts.append(f'<text x="{x:.2f}" y="742" text-anchor="middle" font-family="monospace" font-size="12">C{concurrency}</text>')
    parts.append(f'<text x="{(left + right) / 2}" y="758" text-anchor="middle" font-family="sans-serif" font-size="14">Concurrent requests</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    execution = list(csv.DictReader((root / "execution.tsv").open(encoding="utf-8"), delimiter="\t"))
    require(len(execution) == 5, f"expected 5 blocks, found {len(execution)}")
    require(all(row["status"] == "ok" and row["returncode"] == "0" for row in execution), "one or more blocks failed")

    per_block: list[dict[str, Any]] = []
    plan_hashes: set[str] = set()
    fixture_hashes: set[str] = set()
    output_hashes: dict[int, set[str]] = {value: set() for value in range(1, 33)}
    for row in execution:
        block = int(row["block"])
        case_dir = root / "blocks" / row["case_dir"]
        result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        require(result.get("status") == "ok", f"block {block} result not ok")
        require(len(result.get("phases", [])) == 32, f"block {block} phase count")
        require(result["shutdown_contract"]["diagnostics_after_shutdown"]["driver"]["session_count"] == 0, f"block {block} leaked driver session")
        require(not result["shutdown_contract"]["cleanup_errors"], f"block {block} cleanup errors")
        placement = result["placement_audit"]
        require(placement["selected_expert_count"] == 480, f"block {block} plan count")
        require(placement["selected_parameter_bytes_per_forward"] == 2_264_924_160, f"block {block} selected bytes")
        require(placement["remote_physical_residency"]["cpu_bytes"] == 2_264_924_160, f"block {block} CPU residency")
        require(placement["remote_physical_residency"]["gpu_bytes"] == 0, f"block {block} GPU residency")
        require(placement["remote_physical_residency"]["unknown_bytes"] == 0, f"block {block} unknown residency")
        require(result["final_audit"]["manifest"]["actual_h2d_bytes"] == 0, f"block {block} copied weights")
        require(result["final_audit"]["parameter_va"]["va_sha256"] == placement["parameter_va_after_load"]["va_sha256"], f"block {block} VA changed")
        plan_hashes.add(result["config"]["plan_sha256"])
        fixture_hashes.add(result["config"]["fixture_sha256"])
        nvml = read_nvml(case_dir / "nvml.csv")
        gpm = read_gpm(case_dir / "gpm_dmon.txt")
        sar = read_sar(case_dir / "cpu_sar.txt")
        require(nvml, f"block {block} has no NVML samples")
        for phase in result["phases"]:
            concurrency = int(phase["concurrency"])
            output_hashes[concurrency].add(str(phase["output_token_ids_sha256"]))
            require(phase["preemptions"] == 0 and phase["resumes"] == 0, f"block {block} C{concurrency} preempted")
            start = float(phase["decode_measured_started_epoch_seconds"])
            end = float(phase["decode_measured_finished_epoch_seconds"])
            duration = end - start
            require(duration > 0, f"block {block} C{concurrency} invalid window")
            nv = window_rows(nvml, start, end)
            gp = window_rows(gpm, start, end) if gpm else []
            cp = window_rows(sar, start, end) if sar else []
            rx_kib_s = mean_field(nv, "pcie_rx_kib_s")
            tx_kib_s = mean_field(nv, "pcie_tx_kib_s")
            require(rx_kib_s is not None and tx_kib_s is not None, f"block {block} C{concurrency} missing PCIe")
            measured_steps = int(phase["steady_decode_step_ms"]["count"])
            measured_tokens = concurrency * measured_steps
            observed_rx_bytes = rx_kib_s * 1024.0 * duration
            observed_tx_bytes = tx_kib_s * 1024.0 * duration
            per_block.append({
                "block": block,
                "order": row["order"],
                "concurrency": concurrency,
                "measured_decode_steps": measured_steps,
                "measured_decode_tokens": measured_tokens,
                "measured_window_seconds": duration,
                "steady_decode_step_median_ms": float(phase["steady_decode_step_ms"]["median"]),
                "request_tpot_median_ms": float(phase["request_timing"]["request_tpot_ms"]["median"]),
                "decode_tokens_per_second": float(phase["steady_decode_tokens_per_second"]),
                "pcie_rx_gib_s": rx_kib_s * 1024.0 / (1024.0 ** 3),
                "pcie_tx_gib_s": tx_kib_s * 1024.0 / (1024.0 ** 3),
                "pcie_rx_peak_gib_s": (max_field(nv, "pcie_rx_kib_s") or 0.0) * 1024.0 / (1024.0 ** 3),
                "pcie_rx_pct_gen5_x16": 100.0 * rx_kib_s * 1024.0 / PCIE_GEN5_X16_BYTES_S,
                "observed_pcie_rx_bytes": observed_rx_bytes,
                "observed_pcie_tx_bytes": observed_tx_bytes,
                "observed_pcie_rx_bytes_per_request": observed_rx_bytes / concurrency,
                "observed_pcie_rx_bytes_per_decode_token": observed_rx_bytes / measured_tokens,
                "nvml_gpu_util_pct": mean_field(nv, "gpu_util_pct"),
                "nvml_memory_util_pct": mean_field(nv, "memory_util_pct"),
                "gpm_sm_activity_pct": mean_field(gp, "sm_activity_pct") if gp else None,
                "gpm_sm_occupancy_pct": mean_field(gp, "sm_occupancy_pct") if gp else None,
                "gpm_gpu_dram_activity_pct": mean_field(gp, "gpu_dram_activity_pct") if gp else None,
                "gpm_pcie_rx_gib_s": (mean_field(gp, "pcie_rx_mib_s") / 1024.0) if gp and mean_field(gp, "pcie_rx_mib_s") is not None else None,
                "cpu_busy_pct": mean_field(cp, "cpu_busy_pct") if cp else None,
                "output_token_ids_sha256": phase["output_token_ids_sha256"],
                "nvml_sample_count": len(nv),
                "gpm_sample_count": len(gp),
                "cpu_sample_count": len(cp),
            })

    require(len(plan_hashes) == 1 and len(fixture_hashes) == 1, "inputs differ across blocks")
    require(
        all(len(value) == 64 for hashes in output_hashes.values() for value in hashes),
        "one or more output hashes are malformed",
    )
    fields = list(per_block[0])
    write_csv(analysis / "per_block_metrics.csv", per_block, fields)

    aggregate: list[dict[str, Any]] = []
    rng = random.Random(20260825)
    metric_names = [
        "request_tpot_median_ms", "steady_decode_step_median_ms", "decode_tokens_per_second",
        "pcie_rx_gib_s", "pcie_tx_gib_s", "pcie_rx_pct_gen5_x16",
        "observed_pcie_rx_bytes_per_request", "observed_pcie_rx_bytes_per_decode_token",
        "nvml_gpu_util_pct", "gpm_sm_activity_pct", "gpm_sm_occupancy_pct",
        "gpm_gpu_dram_activity_pct", "cpu_busy_pct",
    ]
    for concurrency in range(1, 33):
        records = [row for row in per_block if row["concurrency"] == concurrency]
        require(len(records) == 5, f"C{concurrency} does not have five blocks")
        output: dict[str, Any] = {"concurrency": concurrency, "block_count": len(records)}
        for key in metric_names:
            values = [float(row[key]) for row in records if row.get(key) is not None]
            output[key] = median(values) if values else None
            if key in {"request_tpot_median_ms", "pcie_rx_gib_s", "gpm_sm_activity_pct"} and values:
                low, high = bootstrap_median_ci(values, rng)
                output[key + "_ci95_low"] = low
                output[key + "_ci95_high"] = high
        aggregate.append(output)
    aggregate_fields = list(aggregate[0])
    write_csv(analysis / "aggregate_metrics.csv", aggregate, aggregate_fields)

    low_concurrency_by_block: dict[int, float] = {}
    for block in range(5):
        low_concurrency_by_block[block] = median([
            float(row["request_tpot_median_ms"])
            for row in per_block
            if row["block"] == block and int(row["concurrency"]) <= 4
        ])
    agreement: dict[int, int] = {}
    for concurrency in range(1, 33):
        agreement[concurrency] = sum(
            1
            for row in per_block
            if row["concurrency"] == concurrency
            and float(row["request_tpot_median_ms"]) >= 1.10 * low_concurrency_by_block[int(row["block"])]
        )
    first: int | None = None
    for concurrency in range(1, 31):
        if all(agreement[value] >= 4 for value in (concurrency, concurrency + 1, concurrency + 2)):
            first = concurrency
            break
    peak_rx = max(float(row["pcie_rx_gib_s"]) for row in aggregate)
    peak_rx_concurrency = max(aggregate, key=lambda row: float(row["pcie_rx_gib_s"]))["concurrency"]
    decision = {
        "schema": "eurosys_e4_degradation_decision_v1",
        "rule": "TPOT >= 1.10x each block's pooled C1-C4 median, agreement in >=4/5 independent process blocks, sustained for three consecutive concurrency levels",
        "low_concurrency_baseline_tpot_ms_by_block": low_concurrency_by_block,
        "agreement_blocks_by_concurrency": agreement,
        "first_sustained_degradation_concurrency": first,
        "remote_read_validated_through_concurrency": 32 if first is None else first - 1,
        "peak_observed_pcie_rx_gib_s": peak_rx,
        "peak_observed_pcie_rx_concurrency": peak_rx_concurrency,
        "peak_observed_pcie_rx_pct_gen5_x16": 100.0 * peak_rx * (1024.0 ** 3) / PCIE_GEN5_X16_BYTES_S,
        "pcie_gen5_x16_theoretical_gib_s": PCIE_GEN5_X16_BYTES_S / (1024.0 ** 3),
        "cpu_dram_bandwidth_status": "N/A",
        "cpu_dram_bandwidth_reason": "KVM does not expose uncore IMC PMUs or server memory-controller PCI configuration; Intel PCM reports no memory controllers",
        "switch_to_copy_cache_recommendation": (
            "no TPOT-defined switch point observed through C32"
            if first is None
            else f"switch before C{first}, subject to copy/cache capacity"
        ),
    }
    (analysis / "degradation_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    svg_plot(analysis / "e4_concurrency_summary.svg", aggregate, decision)
    verification = {
        "status": "ok",
        "schema": "eurosys_e4_verification_v1",
        "process_blocks": 5,
        "concurrency_levels": 32,
        "phase_records": len(per_block),
        "plan_sha256": next(iter(plan_hashes)),
        "fixture_sha256": next(iter(fixture_hashes)),
        "output_hashes_recorded": True,
        "cross_block_output_hash_comparison": (
            "not applicable: independent blocks rotate the frozen prompt list by 0,7,14,21,28"
        ),
        "placement": {
            "remote_experts": 480,
            "remote_bytes": 2_264_924_160,
            "all_cpu_resident": True,
            "unknown_bytes": 0,
            "h2d_copy_bytes": 0,
        },
        "preemptions": 0,
        "driver_sessions_after_shutdown": 0,
        "cpu_dram_bandwidth": "N/A",
        "decision": decision,
    }
    (analysis / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
