#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Plot the historical C16, 512 MiB KV throughput with publication styling.

Run with: uv run experiments/adaptmoe_token_threshold_sharegpt_compare/plot_kv512_qps_throughput.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
QPS = (4, 8, 10, 20, 30)
SERIES = (
    ("adaptive", "AdaptMoE", "#2F4B66", "o", "-"),
    ("native_uva", "Native UVA", "#52665A", "^", (0, (1.2, 1.8))),
    ("native_copy", "Native Copy", "#7A4747", "s", (0, (4, 2))),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=HERE / "KV512_HISTORICAL_RESULTS_20261024.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "figures/kv512_qps_throughput_20261024",
    )
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        selected = [
            row for row in csv.DictReader(stream)
            if row["group"] == "main_n128_c16"
            and row["arm"] in {series[0] for series in SERIES}
        ]
    points = {(int(row["offered_qps"]), row["arm"]): row for row in selected}
    expected = {(qps, arm) for qps in QPS for arm, *_ in SERIES}
    if len(selected) != 15 or set(points) != expected:
        raise ValueError("Expected five QPS values for each of the three mechanisms")
    for row in selected:
        if (int(row["kv_budget_reference_bytes"]) != 536870912
                or int(row["max_num_seqs"]) != 16
                or int(row["requests_per_process"]) != 128
                or int(row["samples"]) != 2):
            raise ValueError("The selected rows do not match the historical protocol")

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.labelcolor": "#252525",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.75,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "kv512-qps-throughput-20261024",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    fig.subplots_adjust(left=0.175, right=0.975, bottom=0.19, top=0.975)

    for arm, label, color, marker, line_style in SERIES:
        ax.plot(
            QPS, [float(points[qps, arm]["actual_qps"]) for qps in QPS],
            label=label, color=color, linestyle=line_style,
            linewidth=1.7, marker=marker, markersize=5.0,
            markeredgewidth=1.15,
            markerfacecolor=color if arm == "adaptive" else "white",
            markeredgecolor=color, clip_on=False,
            zorder=4 if arm == "adaptive" else 3,
        )

    ax.set_xlabel("QPS (requests/s)", labelpad=5)
    ax.set_ylabel("Actual throughput (requests/s)", labelpad=6)
    ax.set_xlim(2.7, 31.3)
    ax.set_ylim(0, 11)
    ax.set_xticks(QPS)
    ax.set_yticks((0, 2, 4, 6, 8, 10))
    ax.tick_params(axis="both", direction="out", length=3, width=0.75, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DADADA", linewidth=0.55, alpha=0.8)
    ax.legend(
        loc="lower right", frameon=False, borderaxespad=0.8,
        handlelength=2.5, handletextpad=0.7, labelspacing=0.5,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "kv512_qps_throughput"
    for extension in ("pdf", "svg", "png"):
        target = stem.with_suffix(f".{extension}")
        fig.savefig(target, dpi=600)
        print(target)
    plt.close(fig)

    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("offered_qps", *(series[1] for series in SERIES)))
        for qps in QPS:
            writer.writerow((qps, *(points[qps, series[0]]["actual_qps"] for series in SERIES)))
    stem.with_suffix(".txt").write_text(
        "Actual throughput versus offered QPS for DeepSeek-V2-Lite-Chat BF16, "
        "with a fixed 512 MiB KV cache and concurrency limit 16. Each run replays "
        "128 ShareGPT requests with 512 input tokens and 16 output tokens. "
        "Points show arithmetic means of two independent processes; straight "
        "segments connect measured points. Throughput includes final drain. "
        "QPS=8 uses the historical selected rerun.\n"
        "Style: dark muted blue (AdaptMoE), green (Native UVA), and red (Native Copy); "
        "distinct line styles and markers support monochrome reproduction. "
        "PDF embeds TrueType fonts; SVG retains editable text; PNG is 600 dpi.\n"
        f"Source: {args.input.resolve()}\n"
        f"Reproduce: uv run {Path(__file__).resolve()}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
