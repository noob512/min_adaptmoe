#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot the E13 Fixed-K2 versus Fixed-K6 low-KV-pressure tradeoff."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_fixed_plan_tradeoff"

DEFAULT_REPORT = (
    REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md"
)
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

BATCH_SIZES = (64, 128)
PLANS = ("Fixed-K2", "Fixed-K6")
STYLES = {
    "Fixed-K2": {"face": "#0072B2", "hatch": "//"},
    "Fixed-K6": {"face": "#E69F00", "hatch": "\\\\"},
}
EDGE_COLOR = "#343434"
CSV_COLUMNS = (
    "model",
    "workload",
    "batch_size",
    "plan",
    "request_throughput_req_s",
    "e2e_latency_ms",
    "sample_nature",
    "source_section",
    "validity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--caption", type=Path, default=DEFAULT_CAPTION)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def extract_section(report: str) -> str:
    start_heading = "### 16.2 Fixed-K2与Fixed-K6"
    end_heading = "### 16.3 Adaptive切档后的稳态等价"
    start = report.find(start_heading)
    end = report.find(end_heading, start + len(start_heading))
    require(start >= 0 and end > start, "E13 §16.2 not found")
    return report[start:end]


def parse_positive(cell: str, label: str) -> float:
    cleaned = cell.strip().replace(",", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?: ms)?", cleaned)
    require(match is not None, f"invalid {label}: {cell}")
    value = float(match.group(1))
    require(value > 0 and math.isfinite(value), f"non-positive {label}: {cell}")
    return value


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    section = extract_section(report)
    require("吞吐分别比K6高15.09%/21.86%" in section, "throughput deltas differ")
    require("E2E分别低13.12%/17.99%" in section, "E2E deltas differ")
    require("不能外推为\n高压力时K2仍优于K6" in section, "scope limitation missing")

    header = ["B", "档位", "req/s", "TTFT", "TPOT", "E2E", "Prefill", "P→D", "steady Decode"]
    lines = section.splitlines()
    start = next((i for i, line in enumerate(lines) if split_markdown_row(line) == header), -1)
    require(start >= 0, "§16.2 table header not found")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    require(len(table_lines) == 6, "§16.2 table row count differs")

    records: list[dict[str, object]] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        require(len(cells) == len(header), f"malformed §16.2 row: {line}")
        row = dict(zip(header, cells, strict=True))
        batch_size = int(row["B"])
        plan = row["档位"]
        require(batch_size in BATCH_SIZES, f"unexpected batch size: {batch_size}")
        require(plan in PLANS, f"unexpected plan: {plan}")
        records.append(
            {
                "model": "Qwen3-30B-A3B FP8",
                "workload": "Short-128 / Output-32",
                "batch_size": batch_size,
                "plan": plan,
                "request_throughput_req_s": parse_positive(row["req/s"], "req/s"),
                "e2e_latency_ms": parse_positive(row["E2E"], "E2E"),
                "sample_nature": "median of three within-process samples after one excluded warmup",
                "source_section": (
                    "RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md §16.2 Fixed-K2 and Fixed-K6"
                ),
                "validity": "valid_single_cold_process_arm",
            }
        )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 4, "expected four Fixed-plan rows")
    require(not data.duplicated(["batch_size", "plan"]).any(), "duplicate Fixed-plan row")
    plan_rank = {plan: i for i, plan in enumerate(PLANS)}
    data["_plan"] = data["plan"].map(plan_rank)
    data = data.sort_values(["batch_size", "_plan"]).drop(columns="_plan")

    indexed = data.set_index(["batch_size", "plan"])
    expected_throughput = {64: 15.09, 128: 21.86}
    expected_e2e = {64: -13.12, 128: -17.99}
    for batch_size in BATCH_SIZES:
        k2 = indexed.loc[(batch_size, "Fixed-K2")]
        k6 = indexed.loc[(batch_size, "Fixed-K6")]
        throughput_delta = (
            float(k2["request_throughput_req_s"]) / float(k6["request_throughput_req_s"]) - 1
        ) * 100
        e2e_delta = (float(k2["e2e_latency_ms"]) / float(k6["e2e_latency_ms"]) - 1) * 100
        require(round(throughput_delta, 2) == expected_throughput[batch_size],
                f"B{batch_size} throughput delta differs")
        require(round(e2e_delta, 2) == expected_e2e[batch_size],
                f"B{batch_size} E2E delta differs")
    return data.loc[:, CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.6f")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == list(CSV_COLUMNS), "CSV columns differ")
    require(len(serialized) == 4, "serialized row count differs")
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.3,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def compute_deltas(data: pd.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    indexed = data.set_index(["batch_size", "plan"])
    throughput: dict[int, float] = {}
    e2e: dict[int, float] = {}
    for batch_size in BATCH_SIZES:
        k2 = indexed.loc[(batch_size, "Fixed-K2")]
        k6 = indexed.loc[(batch_size, "Fixed-K6")]
        throughput[batch_size] = (
            float(k2["request_throughput_req_s"]) / float(k6["request_throughput_req_s"]) - 1
        ) * 100
        e2e[batch_size] = (
            float(k2["e2e_latency_ms"]) / float(k6["e2e_latency_ms"]) - 1
        ) * 100
    return throughput, e2e


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    throughput_deltas, e2e_deltas = compute_deltas(data)
    panels = (
        ("request_throughput_req_s", "(a) Request throughput", "Request throughput (req/s)"),
        ("e2e_latency_ms", "(b) End-to-end latency", "End-to-end latency (ms)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    centers = np.arange(len(BATCH_SIZES), dtype=float)
    width = 0.30

    for panel_index, (axis, (column, title, ylabel)) in enumerate(zip(axes, panels, strict=True)):
        maximum = float(data[column].max())
        for plan_index, plan in enumerate(PLANS):
            selected = data[data["plan"] == plan].sort_values("batch_size")
            x = centers + (-width / 2 if plan_index == 0 else width / 2)
            style = STYLES[plan]
            axis.bar(
                x,
                selected[column],
                width=width,
                color=style["face"],
                edgecolor=EDGE_COLOR,
                linewidth=0.65,
                hatch=style["hatch"],
                label=plan,
                zorder=3,
            )
        deltas = throughput_deltas if panel_index == 0 else e2e_deltas
        for group_index, batch_size in enumerate(BATCH_SIZES):
            group = data[data["batch_size"] == batch_size]
            y = float(group[column].max())
            delta = deltas[batch_size]
            sign = "+" if delta >= 0 else "−"
            axis.annotate(
                f"K2 vs K6: {sign}{abs(delta):.2f}%",
                (group_index, y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                color="#333333",
            )
        axis.set_ylim(0, maximum * 1.28)
        axis.set_xticks(centers, [str(value) for value in BATCH_SIZES])
        axis.set_xlabel("Batch size")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", pad=3.0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DCDCDC", linewidth=0.6)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles = [
        Patch(
            facecolor=STYLES[plan]["face"],
            edgecolor=EDGE_COLOR,
            linewidth=0.65,
            hatch=STYLES[plan]["hatch"],
            label=plan,
        )
        for plan in PLANS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.21, top=0.80, wspace=0.27)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Fixed-K2 versus Fixed-K6 under low KV demand",
        "Subject": "Request throughput and end-to-end latency",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(png_path, format="png", dpi=300, metadata={"Software": SCRIPT_PATH.name})
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "Fixed-plan performance on Qwen3-30B-A3B FP8 with Short-128/Output-32 at "
        "batch sizes 64 and 128. Both workloads remain below Fixed-K2's KV-capacity "
        "limit, so Fixed-K6 provides no admission benefit while offloading more experts. "
        "When KV demand fits in the smaller tier, reserving additional KV capacity "
        "unnecessarily reduces expert residency and serving performance. Each arm is one "
        "cold process and each value is the median of three within-process samples after an "
        "excluded warmup; no error bars are shown. This result covers only the measured "
        "low-KV-pressure coordinates and does not imply that K2 remains preferable under "
        "high KV pressure."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    require(report_path.is_file(), f"report not found: {report_path}")
    data = build_data(report_path)
    plot_data = write_and_reload_csv(data, args.csv.resolve())
    render_figure(plot_data, args.pdf.resolve(), args.png.resolve())
    write_caption(args.caption.resolve())
    print(f"[OK] sole source: {report_path}")
    print(f"[OK] CSV: {args.csv.resolve()} ({len(plot_data)} rows)")
    print(f"[OK] PDF: {args.pdf.resolve()}")
    print(f"[OK] PNG: {args.png.resolve()}")
    print(f"[OK] caption: {args.caption.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
