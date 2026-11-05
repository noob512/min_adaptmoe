#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot E2E latency dumbbells for Stock UVM versus Native UVA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import fig_stock_uvm_vs_native_uva as source


SCRIPT_PATH = Path(__file__).resolve()
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_stock_uvm_vs_native_uva_e2e"

DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

NATIVE_COLOR = "#0072B2"
STOCK_COLOR = "#E69F00"
CONNECTOR_COLOR = "#B9B9B9"

CSV_COLUMNS = [
    "model",
    "workload",
    "concurrency",
    "native_uva_e2e_ms",
    "stock_uvm_e2e_ms",
    "slowdown",
    "source_section",
    "validity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-report", type=Path, default=source.DEFAULT_STOCK_REPORT)
    parser.add_argument("--native-report", type=Path, default=source.DEFAULT_NATIVE_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--caption", type=Path, default=DEFAULT_CAPTION)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_all_ratios(stock_report: Path, native_report: Path) -> pd.DataFrame:
    stock = source.extract_stock_data(stock_report)
    native = source.extract_native_data(native_report)
    return source.merge_sources(stock, native)


def build_e2e_csv_data(all_ratios: pd.DataFrame) -> pd.DataFrame:
    e2e = all_ratios[all_ratios["metric"] == "E2E"].copy()
    e2e = e2e.rename(
        columns={
            "native_uva_ms": "native_uva_e2e_ms",
            "stock_uvm_ms": "stock_uvm_e2e_ms",
            "ratio": "slowdown",
        }
    )
    e2e = e2e.drop(columns=["metric", "log2_ratio"])
    require(len(e2e) == 18, f"expected 18 E2E rows, found {len(e2e)}")
    require(
        (e2e["slowdown"] > 1.0).all(),
        "a Stock UVM E2E value is not to the right of Native UVA",
    )
    return e2e[CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.8f")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == CSV_COLUMNS, "serialized CSV columns differ")
    require(len(serialized) == 18, "serialized CSV row count differs")
    recomputed = serialized["stock_uvm_e2e_ms"] / serialized["native_uva_e2e_ms"]
    require(
        np.allclose(serialized["slowdown"], recomputed, rtol=0, atol=5e-9),
        "serialized slowdowns do not match source E2E milliseconds",
    )
    require(
        set(serialized["validity"]) == {"valid_completed_single_run_unpaired"},
        "unexpected validity value",
    )
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.2,
            "xtick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 0.0,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def row_positions(configs: list[tuple[str, int]]) -> np.ndarray:
    positions: list[float] = []
    current = 0.0
    previous_workload: str | None = None
    for workload, _concurrency in configs:
        if previous_workload is not None and workload != previous_workload:
            current += 0.38
        positions.append(current)
        current += 1.0
        previous_workload = workload
    return np.asarray(positions)


def latency_tick(value: float, _position: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    panel_titles = {
        source.MODELS[0]: "(a) DeepSeek-V2-Lite BF16",
        source.MODELS[1]: "(b) Qwen3-30B-A3B FP8",
    }

    minimum_native = float(data["native_uva_e2e_ms"].min())
    maximum_stock = float(data["stock_uvm_e2e_ms"].max())
    x_min = minimum_native / 1.40
    x_max = maximum_stock * 1.80

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [8.0, 10.0]},
    )
    for axis, model in zip(axes, source.MODELS, strict=True):
        configs = source.CONFIGS[model]
        positions = row_positions(configs)
        panel = data[data["model"] == model].set_index(["workload", "concurrency"])

        for y_position, config in zip(positions, configs, strict=True):
            row = panel.loc[config]
            native_ms = float(row["native_uva_e2e_ms"])
            stock_ms = float(row["stock_uvm_e2e_ms"])
            slowdown = float(row["slowdown"])
            require(stock_ms > native_ms, f"unexpected E2E ordering for {model}/{config}")

            axis.plot(
                [native_ms, stock_ms],
                [y_position, y_position],
                color=CONNECTOR_COLOR,
                linewidth=1.0,
                solid_capstyle="round",
                zorder=2,
            )
            axis.plot(
                native_ms,
                y_position,
                marker="o",
                markersize=5.3,
                markerfacecolor=NATIVE_COLOR,
                markeredgecolor=NATIVE_COLOR,
                linestyle="none",
                zorder=3,
            )
            axis.plot(
                stock_ms,
                y_position,
                marker="s",
                markersize=5.1,
                markerfacecolor=STOCK_COLOR,
                markeredgecolor=STOCK_COLOR,
                linestyle="none",
                zorder=3,
            )
            axis.annotate(
                f"{slowdown:.2f}×",
                xy=(stock_ms, y_position),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.0,
                color="#333333",
                clip_on=False,
                zorder=4,
            )

        labels = [f"{workload} / C{concurrency}" for workload, concurrency in configs]
        axis.set_yticks(positions, labels=labels)
        axis.set_ylim(positions[-1] + 0.55, positions[0] - 0.55)
        axis.set_xscale("log")
        axis.set_xlim(x_min, x_max)
        axis.set_title(panel_titles[model], loc="left", pad=3.5)
        axis.grid(axis="y", color="#EEEEEE", linewidth=0.55, alpha=0.9)
        axis.grid(axis="x", visible=False)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)

    axes[-1].set_xlabel("E2E latency (ms)")
    axes[-1].xaxis.set_major_locator(FixedLocator([1_000, 10_000, 100_000]))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(latency_tick))
    axes[-1].xaxis.set_minor_formatter(NullFormatter())
    figure.supylabel("Workload / Concurrency", x=0.018, fontsize=8.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NATIVE_COLOR,
            markeredgecolor=NATIVE_COLOR,
            markersize=5.3,
            label="Native UVA",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=STOCK_COLOR,
            markeredgecolor=STOCK_COLOR,
            markersize=5.1,
            label="Stock UVM demand paging",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.995),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
    )
    figure.subplots_adjust(left=0.215, right=0.985, bottom=0.105, top=0.925, hspace=0.33)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Stock UVM versus Native UVA end-to-end latency",
        "Subject": "Horizontal dumbbell comparison of E2E latency in milliseconds",
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


def metric_range(all_ratios: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = all_ratios.loc[all_ratios["metric"] == metric, "ratio"]
    return float(values.min()), float(values.max())


def write_caption(all_ratios: pd.DataFrame, path: Path) -> None:
    ttft_min, ttft_max = metric_range(all_ratios, "TTFT")
    tpot_min, tpot_max = metric_range(all_ratios, "TPOT")
    caption = (
        "End-to-end latency of Stock UVM demand paging and Native UVA remote access. "
        "Each horizontal segment compares the two memory paths for the same model, "
        "workload, and concurrency; labels report Stock UVM / Native UVA. Stock UVM "
        "increases end-to-end latency across all evaluated configurations. Values are "
        "request-level P50 E2E latencies in milliseconds for DeepSeek-V2-Lite BF16 and "
        "Qwen3-30B-A3B FP8 at C1 and C16. The DeepSeek Native UVA baseline uses eager "
        "Prefill with a Decode Graph; the Qwen baseline is eager in both phases. Each "
        "source cell is one cold-process result and the runs are unpaired, so no error "
        "bars or confidence intervals are shown. Across these configurations, the "
        f"Stock UVM / Native UVA ratio ranges from {ttft_min:.2f}–{ttft_max:.2f}× for "
        f"TTFT and {tpot_min:.2f}–{tpot_max:.2f}× for TPOT. TTFT has two RAG/C16 "
        "exceptions below 1×, while TPOT is higher with Stock UVM throughout. Results "
        "are limited to one RTX 5090, frozen requests, and the reported cross-run "
        "protocols."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_statistics(data: pd.DataFrame) -> None:
    print(
        "E2E slowdown min/max: "
        f"{data['slowdown'].min():.4f}× / {data['slowdown'].max():.4f}×"
    )
    for model in source.MODELS:
        values = data.loc[data["model"] == model, "slowdown"].to_numpy(dtype=float)
        geometric_mean = float(np.exp(np.log(values).mean()))
        print(f"{model} E2E geometric mean slowdown: {geometric_mean:.4f}×")


def main() -> int:
    args = parse_args()
    stock_report = args.stock_report.resolve()
    native_report = args.native_report.resolve()
    csv_path = args.csv.resolve()
    pdf_path = args.pdf.resolve()
    png_path = args.png.resolve()
    caption_path = args.caption.resolve()
    require(stock_report.is_file(), f"Stock report not found: {stock_report}")
    require(native_report.is_file(), f"Native report not found: {native_report}")

    all_ratios = build_all_ratios(stock_report, native_report)
    e2e_data = build_e2e_csv_data(all_ratios)
    plot_data = write_and_reload_csv(e2e_data, csv_path)
    render_figure(plot_data, pdf_path, png_path)
    write_caption(all_ratios, caption_path)

    print(f"[OK] CSV: {csv_path} ({len(plot_data)} E2E rows)")
    print(f"[OK] PDF: {pdf_path}")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] caption: {caption_path}")
    print_statistics(plot_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
