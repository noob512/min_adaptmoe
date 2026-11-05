#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "numpy>=1.24,<3",
#   "pandas>=2.0,<4",
# ]
# ///
"""Recompose selected annotated figures from the existing scientific CSVs.

The first output replaces the annotated page-table-switch figure with its
original panel (b) and memory-accounting panel (a). The second combines the
filtered Qwen latency panels with both E14 placement-scaling panels. Other
source images and clean variants are not rewritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter
from matplotlib.ticker import StrMethodFormatter

import fig_pd_page_table_switch as switch
import fig_plan_memory_accounting as memory
import fig_qwen_stock_uvm_vs_native_uva as qwen
import plot_fig_e14_placement_scaling as e14
import render_all_dark_variants as variants


FIGURES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "new_output"
PLACEMENT_FIGURE = "fig_qwen_stock_uvm_vs_native_uva_e14_placement_scaling_annotated"


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.55, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_switch(axis: plt.Axes, data: pd.DataFrame) -> None:
    centers = np.arange(len(switch.DIRECTIONS), dtype=float)
    width = 0.30
    for index, method in enumerate(("Parent/PDE", "Leaf-PTE")):
        statistic = "median" if method == "Parent/PDE" else "outer_wall"
        values = [
            switch.value(data, "b", direction, method, statistic)
            for direction in switch.DIRECTIONS
        ]
        style = switch.METHOD_STYLES[method]
        axis.bar(
            centers + (-width / 2 if index == 0 else width / 2), values,
            width=width, color=style["color"], edgecolor="#333333",
            linewidth=0.6, hatch=style["hatch"], label=method, zorder=3,
        )
    for center, direction in zip(centers, switch.DIRECTIONS, strict=True):
        parent = switch.value(data, "b", direction, "Parent/PDE", "median")
        leaf = switch.value(data, "b", direction, "Leaf-PTE", "outer_wall")
        axis.annotate(
            f"{leaf / parent:.1f}×", (center + width / 2, leaf),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=6.85, fontweight="bold", color=switch.METHOD_STYLES["Leaf-PTE"]["color"],
        )
    axis.set_yscale("log")
    axis.set_ylim(15, 650)
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(centers, ["Prefill→\nDecode", "Decode→\nPrefill"])
    axis.set_ylabel("Latency (ms)")
    axis.set_title("(a) Parent vs Leaf-PTE", loc="left", pad=5)
    axis.legend(
        loc="upper left", frameon=False, handlelength=1.4,
        borderaxespad=0.2, fontsize=6.65,
    )
    style_axis(axis)


def draw_memory(axis: plt.Axes, data: pd.DataFrame) -> None:
    centers = np.arange(len(memory.TRANSITIONS), dtype=float)
    width = 0.235
    transition_data = data[data["scope"] == "transition"]
    for index, component in enumerate(memory.COMPONENTS):
        rows = transition_data[transition_data["component"] == component]
        rows = rows.set_index("transition").loc[list(memory.TRANSITIONS)]
        values = rows["change_pages"].to_numpy(dtype=int)
        bars = axis.bar(
            centers + (index - 1) * width, values / 1000.0,
            width=width * 0.90, color=memory.COLORS[component],
            edgecolor=memory.EDGE_COLOR, linewidth=0.7,
            hatch=memory.HATCHES[component], label=component, zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            axis.annotate(
                memory.signed_integer(int(value)),
                (bar.get_x() + bar.get_width() / 2, value / 1000.0),
                xytext=(0, 3 if value >= 0 else -3), textcoords="offset points",
                ha="center", va="bottom" if value >= 0 else "top",
                fontsize=6.5, color="#222222", clip_on=False,
            )
    axis.axhline(0, color="#555555", linewidth=1.0, zorder=2)
    axis.set_xticks(centers, memory.TRANSITIONS)
    axis.set_ylim(-45, 45)
    axis.set_yticks((-40, -20, 0, 20, 40))
    axis.set_ylabel("Change in 64 KiB GPU pages (×10³)")
    axis.set_title("(b) Per-transition accounting", loc="left", pad=5)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=3,
        frameon=False, fontsize=6.4, handlelength=1.3, columnspacing=0.9,
    )
    style_axis(axis)


def draw_qwen(axis: plt.Axes, data: pd.DataFrame, metric: str) -> None:
    centers = np.arange(len(data), dtype=float)
    native = data[f"native_uva_{metric}_ms"].to_numpy(dtype=float)
    stock = data[f"stock_uvm_{metric}_ms"].to_numpy(dtype=float)
    ratios = data[f"{metric}_ratio"].to_numpy(dtype=float)
    if not np.allclose(stock / native, ratios, rtol=0, atol=5e-9):
        raise ValueError(f"Qwen {metric} ratios differ from source milliseconds")
    axis.bar(
        centers - qwen.BAR_OFFSET, native - 1.0, bottom=1.0,
        width=qwen.BAR_WIDTH, color=qwen.NATIVE_COLOR, edgecolor="none", zorder=3,
    )
    axis.bar(
        centers + qwen.BAR_OFFSET, stock - 1.0, bottom=1.0,
        width=qwen.BAR_WIDTH, color=qwen.STOCK_COLOR, edgecolor="none", zorder=3,
    )
    for index, ratio in enumerate(ratios):
        label_x = centers[index] + (
            qwen.BAR_OFFSET if stock[index] >= native[index] else -qwen.BAR_OFFSET
        )
        axis.annotate(
            f"{ratio:.2f}×", (label_x, max(native[index], stock[index])),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=7, color="#333333", clip_on=False,
        )
    for separator in np.arange(len(data) - 1) + 0.5:
        axis.axvline(separator, color="#D8D8D8", linewidth=0.7, zorder=1)
    axis.set_yscale("log")
    axis.set_ylim(min(native.min(), stock.min()) / 2.5, max(native.max(), stock.max()) * 2.5)
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    axis.yaxis.set_major_formatter(FuncFormatter(qwen.latency_tick))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(centers, labels=data["workload"].tolist())
    axis.set_xlabel("Workload")
    axis.set_ylabel("Latency (ms)")
    axis.set_title(metric.upper(), loc="left", pad=4)
    style_axis(axis)


def draw_placement(axis: plt.Axes, aggregate: dict, phase: str) -> None:
    plotted_max = 0.0
    for scheme in e14.SCHEME_ORDER:
        style = e14.scheme_style(scheme)
        medians = [aggregate[(scheme, phase, batch)]["median"] for batch in e14.BATCHES]
        lowers = [
            aggregate[(scheme, phase, batch)]["median"] - aggregate[(scheme, phase, batch)]["min"]
            for batch in e14.BATCHES
        ]
        uppers = [
            aggregate[(scheme, phase, batch)]["max"] - aggregate[(scheme, phase, batch)]["median"]
            for batch in e14.BATCHES
        ]
        # Preserve the existing E14 omission of the Prefill/B16 range stem.
        if phase == "prefill" and scheme == "Layer-wise cold-expert offload":
            lowers[e14.BATCHES.index(16)] = 0.0
            uppers[e14.BATCHES.index(16)] = 0.0
        plotted_max = max(plotted_max, max(np.asarray(medians) + uppers))
        axis.errorbar(
            e14.BATCHES, medians, yerr=[lowers, uppers], color=style["color"],
            linestyle="-", linewidth=1.25, marker=style["marker"], markersize=5.2,
            markerfacecolor=style["marker_face"], markeredgecolor=style["color"],
            markeredgewidth=1.05, elinewidth=1.0, capsize=0.0, zorder=3,
        )
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.78, 41)
    axis.set_xticks(e14.BATCHES, labels=[str(batch) for batch in e14.BATCHES])
    axis.set_xlabel("Batch size (requests)")
    axis.set_ylabel("Throughput (tokens/s)")
    axis.set_title(phase.capitalize(), loc="left", pad=4)
    axis.set_ylim(0, plotted_max * 1.12)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    style_axis(axis)


def save_annotated(figure: Figure, output_root: Path, name: str) -> None:
    variants._darken_figure(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    # Catch clipped labels before exporting the vector PDF and 300-dpi PNG.
    for text in figure.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bounds = text.get_window_extent(renderer)
        if (bounds.x0 < -1 or bounds.y0 < -1
                or bounds.x1 > figure.bbox.width + 1
                or bounds.y1 > figure.bbox.height + 1):
            raise ValueError(f"text falls outside {name}: {text.get_text()!r}")
    for extension in ("pdf", "png"):
        path = output_root / extension / f"{name}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = (
            {"Title": name, "Creator": Path(__file__).name}
            if extension == "pdf" else {"Software": Path(__file__).name}
        )
        figure.savefig(path, format=extension, dpi=300, metadata=metadata)
        print(f"[OK] {path}")
    plt.close(figure)


def render_switch_memory(output_root: Path) -> None:
    switch_data = pd.read_csv(FIGURES_ROOT / "data/fig_pd_page_table_switch.csv")
    memory_data = pd.read_csv(FIGURES_ROOT / "data/fig_plan_memory_accounting.csv")
    figure, axes = plt.subplots(
        1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": (1.0, 1.7)},
    )
    draw_switch(axes[0], switch_data)
    draw_memory(axes[1], memory_data)
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.76, wspace=0.43)
    assert len(axes[0].patches) == 4 and len(axes[1].patches) == 12
    save_annotated(figure, output_root, "fig_pd_page_table_switch_annotated")


def render_qwen_placement(output_root: Path) -> None:
    data = pd.read_csv(FIGURES_ROOT / "data/fig_qwen_stock_uvm_vs_native_uva.csv")
    data = data[data["concurrency"] == 1].set_index("workload")
    data = data.loc[["Short", "Medium", "Long"]].reset_index()
    aggregate = e14.aggregate_tidy(e14.read_tidy_csv(FIGURES_ROOT / "data/fig_e14_placement_scaling.csv"))
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 4.45))
    draw_qwen(axes[0, 0], data, "ttft")
    draw_qwen(axes[1, 0], data, "tpot")
    draw_placement(axes[0, 1], aggregate, "prefill")
    draw_placement(axes[1, 1], aggregate, "decode")
    figure.text(0.28, 0.985, "(a) Stock UVM vs Native UVA", ha="center", va="top", fontsize=8.4)
    figure.text(0.79, 0.985, "(b) Offload placement scaling", ha="center", va="top", fontsize=8.4)
    figure.legend(
        handles=[
            Patch(facecolor=qwen.NATIVE_COLOR, edgecolor="none", label="Native UVA"),
            Patch(facecolor=qwen.STOCK_COLOR, edgecolor="none", label="Stock UVM demand paging"),
        ], loc="upper center", bbox_to_anchor=(0.28, 0.94), frameon=False,
        fontsize=7.1, handlelength=1.5, labelspacing=0.5,
    )
    figure.legend(
        handles=[
            Line2D(
                [0], [0], color=e14.scheme_style(scheme)["color"],
                marker=e14.scheme_style(scheme)["marker"],
                markerfacecolor=e14.scheme_style(scheme)["marker_face"],
                markeredgecolor=e14.scheme_style(scheme)["color"],
                markeredgewidth=1.05, linewidth=1.25, markersize=5.2, label=scheme,
            ) for scheme in e14.SCHEME_ORDER
        ], loc="upper center", bbox_to_anchor=(0.79, 0.94), frameon=False,
        fontsize=7.1, handlelength=2.4, labelspacing=0.5,
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.76, wspace=0.50, hspace=0.76)
    assert len(axes[0, 0].patches) == len(axes[1, 0].patches) == 6
    assert [text.get_text() for text in axes[1, 0].get_xticklabels()] == ["Short", "Medium", "Long"]
    save_annotated(figure, output_root, PLACEMENT_FIGURE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    memory.configure_matplotlib()
    matplotlib.rcParams.update({
        "font.size": 7.8, "axes.titlesize": 8.2, "axes.labelsize": 7.6,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    })
    render_switch_memory(args.output_root.resolve())
    render_qwen_placement(args.output_root.resolve())


if __name__ == "__main__":
    main()
