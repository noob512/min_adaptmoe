#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Render the latest six-system E2E matrix as normalized grouped bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

import fig_e2e_six_systems_normalized as source


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_e2e_six_systems_normalized_bars"

DEFAULT_REPORT = REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E14_CONCISE_Full.md"
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

# Family members remain adjacent. Hollow bars denote eager/baseline paths.
BAR_ORDER = (
    "AdaptMoE",
    "Native UVA",
    "UVA Dual-Graph",
    "Native Copy",
    "Copy Dual-Graph",
    "KTransformers",
)
BAR_OFFSETS = dict(zip(BAR_ORDER, np.linspace(-0.325, 0.325, len(BAR_ORDER)), strict=True))
BAR_HEIGHT = 0.105


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--caption", type=Path, default=DEFAULT_CAPTION)
    return parser.parse_args()


def ratio_tick(value: float, _position: int) -> str:
    return f"{value:g}×"


def render_figure(data, pdf_path: Path, png_path: Path) -> None:
    source.configure_matplotlib()
    valid = data.loc[data["status"] == "ok", "normalized_e2e"].astype(float)
    maximum = float(valid.max())
    x_max = max(8.35, np.ceil((maximum + 0.45) * 2.0) / 2.0)
    oom_x = x_max - 0.18

    panels = (
        ("DeepSeek-V2-Lite BF16", 1, "(a) DeepSeek-V2-Lite BF16, C1"),
        ("DeepSeek-V2-Lite BF16", 16, "(b) DeepSeek-V2-Lite BF16, C16"),
        ("Qwen3-30B-A3B FP8", 1, "(c) Qwen3-30B-A3B FP8, C1"),
        ("Qwen3-30B-A3B FP8", 16, "(d) Qwen3-30B-A3B FP8, C16"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)

    for panel_index, (axis, (model, concurrency, title)) in enumerate(
        zip(axes.flat, panels, strict=True)
    ):
        panel = source.panel_data(data, model, concurrency)
        workloads = source.MODEL_WORKLOADS[model]
        bases = np.arange(len(workloads), dtype=float)

        for boundary in np.arange(len(workloads) - 1, dtype=float) + 0.5:
            axis.axhline(boundary, color="#E5E5E5", linewidth=0.55, zorder=0)
        axis.axvline(1.0, color="#6F8291", linewidth=1.15, linestyle="--", zorder=1)

        for workload_index, workload in enumerate(workloads):
            for mechanism in BAR_ORDER:
                row = source.lookup(panel, workload, mechanism)
                y = workload_index + BAR_OFFSETS[mechanism]
                style = source.STYLES[mechanism]

                if row["status"] == "OOM":
                    axis.scatter(
                        [oom_x], [y], marker="s", s=24,
                        facecolors=style["color"], edgecolors=style["color"],
                        linewidths=0.9, zorder=5,
                    )
                    axis.scatter(
                        [oom_x], [y], marker="x", s=34, color="#202020",
                        linewidths=1.0, zorder=6,
                    )
                    axis.annotate(
                        "OOM", (oom_x, y), xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=6.3,
                        fontweight="bold", color="#333333",
                    )
                    continue

                ratio = float(row["normalized_e2e"])
                facecolor = style["color"] if style["filled"] else "white"
                axis.barh(
                    y, ratio, height=BAR_HEIGHT, left=0,
                    facecolor=facecolor, edgecolor=style["color"],
                    linewidth=0.9, zorder=3,
                )

                if bool(row["is_ratio_below_one"]) or bool(
                    row["is_panel_slowest_valid_baseline"]
                ):
                    axis.annotate(
                        f"{ratio:.2f}×", (ratio, y), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=6.2, fontweight="bold", color=style["color"],
                        zorder=7,
                    )

        axis.set_xlim(0, x_max)
        axis.xaxis.set_major_locator(MultipleLocator(2.0))
        axis.xaxis.set_major_formatter(FuncFormatter(ratio_tick))
        axis.set_ylim(len(workloads) - 0.55, -0.55)
        axis.set_yticks(bases)
        if panel_index % 2 == 0:
            axis.set_yticklabels(workloads)
            axis.set_ylabel("Workload")
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", left=False)
        if panel_index >= 2:
            axis.set_xlabel("Normalized E2E latency\n(method / AdaptMoE)")
        else:
            axis.tick_params(axis="x", labelbottom=False)
        axis.set_title(title, loc="left", pad=3.0)
        axis.text(
            0.985, 0.965, "Lower is better", transform=axis.transAxes,
            ha="right", va="top", fontsize=6.0, fontstyle="italic", color="#555555",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    legend_order = (
        "AdaptMoE", "Native UVA", "Native Copy",
        "KTransformers", "UVA Dual-Graph", "Copy Dual-Graph",
    )
    handles = []
    for mechanism in legend_order:
        style = source.STYLES[mechanism]
        handles.append(
            Patch(
                facecolor=style["color"] if style["filled"] else "white",
                edgecolor=style["color"], linewidth=1.0, label=mechanism,
            )
        )
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, handlelength=1.3, handleheight=0.75,
        columnspacing=1.3,
    )
    figure.subplots_adjust(
        left=0.11, right=0.985, bottom=0.13, top=0.84,
        wspace=0.17, hspace=0.31,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Normalized end-to-end latency across six execution configurations",
        "Subject": "Grouped bars from the latest E0--E14 result matrix",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(
        png_path, format="png", dpi=300,
        metadata={"Software": SCRIPT_PATH.name},
    )
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "End-to-end latency normalized to AdaptMoE across six execution configurations. "
        "Each horizontal bar reports one configuration's E2E latency divided by AdaptMoE "
        "for the same model, workload, and concurrency; lower is better, and the dashed "
        "line marks 1×. Hollow and filled bars distinguish eager/baseline and dual-Graph "
        "variants of the same data path. Measurements are single-run observations from "
        "the RESULTS_E0_E14_CONCISE_Full.md primary tables, so no error bars are shown. "
        "OOM is not numericized, and offload placements are not identical for every "
        "configuration."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    source.require(report_path.is_file(), f"report not found: {report_path}")

    data = source.build_data(report_path)
    plot_data = source.write_and_reload_csv(data, args.csv.resolve())
    render_figure(plot_data, args.pdf.resolve(), args.png.resolve())
    write_caption(args.caption.resolve())

    print(f"[OK] source: {report_path}")
    print(f"[OK] CSV: {args.csv.resolve()} ({len(plot_data)} rows)")
    print(f"[OK] PDF: {args.pdf.resolve()}")
    print(f"[OK] PNG: {args.png.resolve()}")
    print(f"[OK] caption: {args.caption.resolve()}")
    source.print_summary(plot_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
