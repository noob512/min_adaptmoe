#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot E13 Adaptive-to-Fixed steady-state ratios as a forest plot."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_adaptive_steady_state"

DEFAULT_REPORT = (
    REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md"
)
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

STATE_MAP = {
    "初始K2": "Initial K2",
    "增长后K6": "Post-grow K6",
    "回缩后K2": "Post-shrink K2",
}
STATE_ORDER = (
    ("Initial K2", 64),
    ("Post-grow K6", 64),
    ("Post-shrink K2", 64),
    ("Initial K2", 128),
    ("Post-grow K6", 128),
    ("Post-shrink K2", 128),
)
METRIC_MAP = {
    "req/s": "Request throughput",
    "Prefill": "Prefill latency",
    "steady Decode": "Steady decode latency",
    "E2E": "E2E latency",
}
METRICS = tuple(METRIC_MAP.values())
STYLES = {
    "Request throughput": {"color": "#0072B2", "marker": "o"},
    "Prefill latency": {"color": "#E69F00", "marker": "s"},
    "Steady decode latency": {"color": "#009E73", "marker": "D"},
    "E2E latency": {"color": "#7E57C2", "marker": "^"},
}
Y_OFFSETS = {
    "Request throughput": -0.24,
    "Prefill latency": -0.08,
    "Steady decode latency": 0.08,
    "E2E latency": 0.24,
}
CSV_COLUMNS = (
    "model",
    "workload",
    "state",
    "corresponding_fixed_plan",
    "batch_size",
    "metric",
    "adaptive_fixed_ratio",
    "within_preregistered_10pct",
    "plan_transaction_excluded",
    "target_tier_warmup_excluded",
    "adaptive_precaptured_graph_entries",
    "full_graph_replay_steps",
    "total_serving_steps",
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


def extract_e13(report: str) -> str:
    start_heading = "## 16. E13：双阶段 CUDA Graph 下的 KV 档位与 Adaptive 回环"
    end_heading = "## 17. E14：同请求集冷专家与固定整层卸载的 eager UVA 扩展性"
    start = report.find(start_heading)
    end = report.find(end_heading, start + len(start_heading))
    require(start >= 0 and end > start, "E13 section not found")
    return report[start:end]


def extract_steady_section(e13: str) -> str:
    start_heading = "### 16.3 Adaptive切档后的稳态等价"
    start = e13.find(start_heading)
    require(start >= 0, "E13 §16.3 not found")
    return e13[start:]


def parse_ratio(cell: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)×", cell.strip())
    require(match is not None, f"invalid Adaptive/Fixed ratio: {cell}")
    value = float(match.group(1))
    require(value > 0 and math.isfinite(value), f"invalid ratio: {cell}")
    return value


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    e13 = extract_e13(report)
    section = extract_steady_section(e13)
    require("Adaptive预捕获cold70/cold80/cold89三档共9个\nGraph entry" in e13,
            "Adaptive Graph-entry count differs")
    require("975/975个step均为FULL Graph replay" in e13, "FULL Graph replay count differs")
    require("捕获后没有Graph重捕获或eager fallback" in e13, "recapture/fallback statement differs")
    require("档位事务和目标档warmup已经排除" in section, "timing exclusion differs")
    require("共24/24个A/F倍率均位于预注册的±10%\n带内" in section,
            "pre-registered-band statement differs")
    require("不能把±10%门写成统计等价检验" in section, "statistical limitation missing")

    header = ["Adaptive状态", "B", "req/s", "Prefill", "steady Decode", "E2E"]
    lines = section.splitlines()
    start = next((i for i, line in enumerate(lines) if split_markdown_row(line) == header), -1)
    require(start >= 0, "§16.3 table header not found")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    require(len(table_lines) == 8, "§16.3 table row count differs")

    records: list[dict[str, object]] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        require(len(cells) == len(header), f"malformed §16.3 row: {line}")
        row = dict(zip(header, cells, strict=True))
        source_state = row["Adaptive状态"]
        require(source_state in STATE_MAP, f"unexpected Adaptive state: {source_state}")
        state = STATE_MAP[source_state]
        batch_size = int(row["B"])
        require((state, batch_size) in STATE_ORDER, f"unexpected state/B: {state}/B{batch_size}")
        fixed_plan = "Fixed-K6" if state == "Post-grow K6" else "Fixed-K2"
        for source_metric, metric in METRIC_MAP.items():
            ratio = parse_ratio(row[source_metric])
            records.append(
                {
                    "model": "Qwen3-30B-A3B FP8",
                    "workload": "Short-128 / Output-32",
                    "state": state,
                    "corresponding_fixed_plan": fixed_plan,
                    "batch_size": batch_size,
                    "metric": metric,
                    "adaptive_fixed_ratio": ratio,
                    "within_preregistered_10pct": 0.9 <= ratio <= 1.1,
                    "plan_transaction_excluded": True,
                    "target_tier_warmup_excluded": True,
                    "adaptive_precaptured_graph_entries": 9,
                    "full_graph_replay_steps": 975,
                    "total_serving_steps": 975,
                    "sample_nature": "median of three within-process samples in one cold-process arm",
                    "source_section": (
                        "RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md "
                        "§16.3 Adaptive steady state after tier changes"
                    ),
                    "validity": "valid_protocol_core_metric_not_statistical_equivalence",
                }
            )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 24, "expected 24 Adaptive/Fixed ratios")
    require(data["within_preregistered_10pct"].all(), "ratio outside pre-registered band")
    require(not data.duplicated(["state", "batch_size", "metric"]).any(), "duplicate ratio")
    state_rank = {state: i for i, state in enumerate(STATE_ORDER)}
    metric_rank = {metric: i for i, metric in enumerate(METRICS)}
    data["_state"] = [state_rank[(state, batch)] for state, batch in zip(data["state"], data["batch_size"])]
    data["_metric"] = data["metric"].map(metric_rank)
    data = data.sort_values(["_state", "_metric"]).drop(columns=["_state", "_metric"])
    return data.loc[:, CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.3f")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == list(CSV_COLUMNS), "CSV columns differ")
    require(len(serialized) == 24, "serialized ratio count differs")
    require(serialized["within_preregistered_10pct"].all(), "serialized band result differs")
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def ratio_tick(value: float, _position: int) -> str:
    return f"{value:.2f}×"


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(7.0, 3.25))
    base_y = np.arange(len(STATE_ORDER), dtype=float)
    labels = [f"{state}, B{batch_size}" for state, batch_size in STATE_ORDER]

    axis.axvspan(0.9, 1.1, color="#DDEEEE", alpha=0.72, linewidth=0, zorder=0)
    axis.axvline(1.0, color="#526D82", linewidth=1.15, zorder=1)
    for boundary in np.arange(len(STATE_ORDER) - 1, dtype=float) + 0.5:
        axis.axhline(boundary, color="#E1E1E1", linewidth=0.55, zorder=1)
    axis.axhline(2.5, color="#B7B7B7", linewidth=0.9, zorder=1)

    for metric in METRICS:
        style = STYLES[metric]
        xs: list[float] = []
        ys: list[float] = []
        for row_index, (state, batch_size) in enumerate(STATE_ORDER):
            selected = data[
                (data["state"] == state)
                & (data["batch_size"] == batch_size)
                & (data["metric"] == metric)
            ]
            require(len(selected) == 1, f"ratio lookup differs: {state}/B{batch_size}/{metric}")
            xs.append(float(selected.iloc[0]["adaptive_fixed_ratio"]))
            ys.append(row_index + Y_OFFSETS[metric])
        axis.scatter(
            xs,
            ys,
            marker=style["marker"],
            s=30,
            facecolors=style["color"],
            edgecolors="white",
            linewidths=0.55,
            label=metric,
            zorder=4,
        )

    axis.set_xlim(0.89, 1.11)
    axis.set_ylim(len(STATE_ORDER) - 0.55, -0.55)
    axis.set_yticks(base_y, labels)
    axis.xaxis.set_major_locator(MultipleLocator(0.05))
    axis.xaxis.set_major_formatter(FuncFormatter(ratio_tick))
    axis.set_xlabel("Adaptive / corresponding Fixed")
    axis.set_ylabel("Adaptive state")
    axis.grid(axis="x", color="#D8D8D8", linewidth=0.55, zorder=0)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=STYLES[metric]["marker"],
            markersize=5.2,
            markerfacecolor=STYLES[metric]["color"],
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=metric,
        )
        for metric in METRICS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    figure.subplots_adjust(left=0.22, right=0.985, bottom=0.19, top=0.82)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Adaptive steady-state performance relative to corresponding Fixed plans",
        "Subject": "Twenty-four pre-registered core metric ratios after tier changes",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(png_path, format="png", dpi=300, metadata={"Software": SCRIPT_PATH.name})
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "Adaptive steady-state performance relative to the corresponding Fixed plan on "
        "Qwen3-30B-A3B FP8 with Short-128/Output-32. Plan transactions and one target-tier "
        "warmup are excluded. All 24/24 pre-registered core-metric ratios lie within the "
        "±10% interval; this is a protocol acceptance result, not statistical equivalence. "
        "Adaptive pre-captured nine Graph entries across three tiers, and 975/975 serving "
        "steps used FULL Graph replay without recapture or eager fallback. After a resize "
        "and one target-tier warmup, AdaptMoE returns to the steady-state performance of "
        "the corresponding fixed plan. Each cell is the median of three within-process "
        "samples from one cold-process arm. The figure excludes switching cost and does not "
        "claim that Adaptive and Fixed are equivalent when that cost is included."
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
    minimum = float(plot_data["adaptive_fixed_ratio"].min())
    maximum = float(plot_data["adaptive_fixed_ratio"].max())
    print(f"[OK] sole source: {report_path}")
    print(f"[OK] CSV: {args.csv.resolve()} ({len(plot_data)} rows)")
    print(f"[OK] PDF: {args.pdf.resolve()}")
    print(f"[OK] PNG: {args.png.resolve()}")
    print(f"[OK] caption: {args.caption.resolve()}")
    print(f"[SUMMARY] 24/24 within ±10%; range {minimum:.3f}×–{maximum:.3f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
