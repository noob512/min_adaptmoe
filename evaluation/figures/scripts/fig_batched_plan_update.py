#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot E6 Plan Class transition latency from the consolidated Markdown report."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_batched_plan_update"

DEFAULT_REPORT = (
    REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md"
)
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

DIFFERENCES = (5, 10, 25, 50, 100)
MECHANISMS = ("Full rebuild", "Serial wave", "Batched difference")
DIRECTIONS = ("Outbound", "Return")
STYLES = {
    "Full rebuild": {
        "color": "#D55E00",
        "marker": "o",
        "linestyle": "-",
    },
    "Serial wave": {
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "--",
    },
    "Batched difference": {
        "color": "#0072B2",
        "marker": "D",
        "linestyle": "-.",
    },
}
GRAPH_CAPTURES = {
    "Full rebuild": (2, 22),
    "Serial wave": (2, 2),
    "Batched difference": (2, 2),
}
CSV_COLUMNS = (
    "changed_experts_pct",
    "direction",
    "mechanism",
    "transition_latency_s",
    "graph_captures_before",
    "graph_captures_after",
    "sample_n",
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


def extract_e6(report: str) -> str:
    heading = "## 9. E6：Plan Class与切换摊销"
    next_heading = "## 10. E7：专家热度与workload shift"
    start = report.find(heading)
    end = report.find(next_heading, start + len(heading))
    require(start >= 0 and end > start, "E6 section not found")
    return report[start:end]


def parse_latency_pair(cell: str) -> tuple[float, float]:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?) s", cell.strip())
    require(match is not None, f"invalid outbound/return cell: {cell}")
    outbound, returned = (float(match.group(1)), float(match.group(2)))
    require(
        outbound > 0 and returned > 0 and math.isfinite(outbound) and math.isfinite(returned),
        f"invalid latency pair: {cell}",
    )
    return outbound, returned


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    section = extract_e6(report)
    require("表中为outbound/return" in section, "direction definition missing")
    require("Graph capture count从\n2增加到22" in section, "Full rebuild capture count differs")
    require("serial和batched均保持2" in section, "Serial/Batched capture count differs")
    require("这是单进程、单样本qualification" in section, "sample nature differs")

    expected_header = ["Difference", *MECHANISMS]
    lines = section.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if split_markdown_row(line) == expected_header),
        -1,
    )
    require(start >= 0, "E6 table header not found")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    require(len(table_lines) == len(DIFFERENCES) + 2, "E6 table row count differs")

    records: list[dict[str, object]] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        require(len(cells) == len(expected_header), f"malformed E6 row: {line}")
        difference_match = re.fullmatch(r"(\d+)%", cells[0])
        require(difference_match is not None, f"invalid Difference value: {cells[0]}")
        difference = int(difference_match.group(1))
        require(difference in DIFFERENCES, f"unexpected Difference: {difference}")
        for mechanism, cell in zip(MECHANISMS, cells[1:], strict=True):
            outbound, returned = parse_latency_pair(cell)
            before, after = GRAPH_CAPTURES[mechanism]
            for direction, latency in zip(DIRECTIONS, (outbound, returned), strict=True):
                records.append(
                    {
                        "changed_experts_pct": difference,
                        "direction": direction,
                        "mechanism": mechanism,
                        "transition_latency_s": latency,
                        "graph_captures_before": before,
                        "graph_captures_after": after,
                        "sample_n": 1,
                        "sample_nature": "single-process directional observation",
                        "source_section": (
                            "RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md "
                            "§9 E6 Plan Class and transition amortization"
                        ),
                        "validity": "valid_hash_matched_single_sample",
                    }
                )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 30, "expected 30 tidy observations")
    require(set(data["changed_experts_pct"]) == set(DIFFERENCES), "Difference coverage differs")
    require(set(data["mechanism"]) == set(MECHANISMS), "mechanism coverage differs")
    require(set(data["direction"]) == set(DIRECTIONS), "direction coverage differs")
    require(
        not data.duplicated(["changed_experts_pct", "direction", "mechanism"]).any(),
        "duplicate observation",
    )
    direction_rank = {direction: index for index, direction in enumerate(DIRECTIONS)}
    mechanism_rank = {mechanism: index for index, mechanism in enumerate(MECHANISMS)}
    data["_direction"] = data["direction"].map(direction_rank)
    data["_mechanism"] = data["mechanism"].map(mechanism_rank)
    data = data.sort_values(
        ["_direction", "_mechanism", "changed_experts_pct"]
    ).drop(columns=["_direction", "_mechanism"])
    return data.loc[:, CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.6f")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == list(CSV_COLUMNS), "CSV columns differ")
    require(len(serialized) == 30, "serialized row count differs")
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.4,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def select_series(data: pd.DataFrame, direction: str, mechanism: str) -> pd.DataFrame:
    selected = data[(data["direction"] == direction) & (data["mechanism"] == mechanism)]
    require(len(selected) == len(DIFFERENCES), f"series coverage differs: {direction}/{mechanism}")
    return selected.sort_values("changed_experts_pct")


def compute_speedups(data: pd.DataFrame) -> dict[str, tuple[float, float]]:
    indexed = data.set_index(["changed_experts_pct", "direction", "mechanism"])
    full_over_batched: list[float] = []
    serial_over_batched: list[float] = []
    for difference in DIFFERENCES:
        for direction in DIRECTIONS:
            full = float(indexed.loc[(difference, direction, "Full rebuild"), "transition_latency_s"])
            serial = float(indexed.loc[(difference, direction, "Serial wave"), "transition_latency_s"])
            batched = float(indexed.loc[(difference, direction, "Batched difference"), "transition_latency_s"])
            full_over_batched.append(full / batched)
            if difference >= 10:
                serial_over_batched.append(serial / batched)
    return {
        "full": (min(full_over_batched), max(full_over_batched)),
        "serial": (min(serial_over_batched), max(serial_over_batched)),
    }


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    speedups = compute_speedups(data)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True)
    panels = (
        ("Outbound", "(a) Outbound transition"),
        ("Return", "(b) Return transition"),
    )

    for panel_index, (axis, (direction, title)) in enumerate(zip(axes, panels, strict=True)):
        for mechanism in MECHANISMS:
            series = select_series(data, direction, mechanism)
            style = STYLES[mechanism]
            axis.plot(
                series["changed_experts_pct"],
                series["transition_latency_s"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.25,
                markersize=4.8,
                markerfacecolor="white" if mechanism == "Serial wave" else style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=1.0,
                zorder=3,
            )
            endpoint = series[series["changed_experts_pct"] == 100].iloc[0]
            value = float(endpoint["transition_latency_s"])
            y_offset = 4 if mechanism == "Serial wave" else 0
            axis.annotate(
                f"{value:.3f}",
                (100, value),
                xytext=(4, y_offset),
                textcoords="offset points",
                ha="left",
                va="bottom" if y_offset else "center",
                fontsize=6.4,
                fontweight="bold",
                color=style["color"],
                clip_on=False,
            )

        axis.set_xlim(0, 111)
        axis.set_ylim(0, 42)
        axis.set_xticks(DIFFERENCES)
        axis.yaxis.set_major_locator(MultipleLocator(10))
        axis.set_xlabel("Changed experts (%)")
        axis.set_title(title, loc="left", pad=3.0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DCDCDC", linewidth=0.6)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if panel_index == 0:
            axis.set_ylabel("Transition latency (s)")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[mechanism]["color"],
            marker=STYLES[mechanism]["marker"],
            linestyle=STYLES[mechanism]["linestyle"],
            linewidth=1.25,
            markersize=4.8,
            markerfacecolor=("white" if mechanism == "Serial wave" else STYLES[mechanism]["color"]),
            markeredgecolor=STYLES[mechanism]["color"],
            markeredgewidth=1.0,
            label=mechanism,
        )
        for mechanism in MECHANISMS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )
    figure.subplots_adjust(left=0.085, right=0.94, bottom=0.20, top=0.80, wspace=0.13)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Plan Class transition amortization",
        "Subject": "Outbound and return transition latency",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(
        png_path,
        format="png",
        dpi=300,
        metadata={"Software": SCRIPT_PATH.name},
    )
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "Plan Class transition latency as the fraction of changed experts increases. "
        "Outbound and return are two directions from the same single-process, single-sample "
        "qualification and are not independent repeats; therefore no error bars are shown. "
        "Batching the placement difference removes repeated synchronization and scales better "
        "than serial waves as the changed set grows, while reusing existing CUDA Graphs. All "
        "three mechanisms produced matching output hashes."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_summary(data: pd.DataFrame) -> None:
    speedups = compute_speedups(data)
    print(
        f"[SUMMARY] Batched vs Full: {speedups['full'][0]:.4f}×–"
        f"{speedups['full'][1]:.4f}×"
    )
    print(
        f"[SUMMARY] Batched vs Serial (10%–100%): {speedups['serial'][0]:.4f}×–"
        f"{speedups['serial'][1]:.4f}×"
    )


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    csv_path = args.csv.resolve()
    pdf_path = args.pdf.resolve()
    png_path = args.png.resolve()
    caption_path = args.caption.resolve()
    require(report_path.is_file(), f"report not found: {report_path}")

    data = build_data(report_path)
    plot_data = write_and_reload_csv(data, csv_path)
    render_figure(plot_data, pdf_path, png_path)
    write_caption(caption_path)

    print(f"[OK] sole source: {report_path}")
    print(f"[OK] CSV: {csv_path} ({len(plot_data)} rows)")
    print(f"[OK] PDF: {pdf_path}")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] caption: {caption_path}")
    print_summary(plot_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
