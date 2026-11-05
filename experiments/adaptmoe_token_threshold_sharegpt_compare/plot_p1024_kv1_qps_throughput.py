#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Plot four mechanisms' actual throughput with solid lines and a top-right legend."""

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
    ("native_uva", "UVA", "#52665A", "^"),
    ("native_copy", "Native Copy", "#7A4747", "s"),
    ("adaptive_fixed_1g", "AdaptMoE (fixed)", "#67576E", "o"),
    ("adaptive_dynamic", "AdaptMoE (dynamic)", "#2F4B66", "D"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=HERE / "P1024_KV1_QPS4_8_10_20_30_FOUR_MECHANISMS_20261024.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "figures/p1024_kv1_qps_throughput_20261024",
    )
    args = parser.parse_args()
    arms = {series[0] for series in SERIES}
    with args.input.open(newline="") as stream:
        selected = [row for row in csv.DictReader(stream)
                    if row["arm"] in arms and float(row["offered_qps"]) in QPS]
    points = {(int(float(row["offered_qps"])), row["arm"]): row for row in selected}
    if len(selected) != 20 or set(points) != {(q, arm) for q in QPS for arm in arms}:
        raise ValueError("Expected exactly five QPS values for each of four mechanisms")

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
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "p1024-kv1-qps-throughput-20261024",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.19, top=0.975)
    for arm, label, color, marker in SERIES:
        dynamic = arm == "adaptive_dynamic"
        ax.plot(
            QPS, [float(points[q, arm]["actual_qps"]) for q in QPS],
            label=label, color=color, linestyle="-", linewidth=1.7,
            marker=marker, markersize=4.6 if dynamic else 5.2,
            markerfacecolor=color if dynamic else "white",
            markeredgecolor=color, markeredgewidth=1.15,
            zorder=5 if dynamic else 4 if arm == "adaptive_fixed_1g" else 3,
        )

    ax.set_xlabel("QPS (requests/s)", labelpad=5)
    ax.set_ylabel("Actual throughput (requests/s)", labelpad=6)
    ax.set_xlim(2.7, 31.3)
    # Reserve space above the measured curves for the requested top-right legend.
    ax.set_ylim(0, 20)
    ax.set_xticks(QPS)
    ax.set_yticks((0, 4, 8, 12, 16, 20))
    ax.tick_params(axis="both", direction="out", length=3, width=0.75, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DADADA", linewidth=0.55, alpha=0.8)
    ax.legend(
        loc="upper right", ncol=2, frameon=False, borderaxespad=0.65,
        handlelength=2.0, handletextpad=0.55,
        columnspacing=1.2, labelspacing=0.55,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "p1024_kv1_qps_throughput"
    for extension in ("pdf", "svg", "png"):
        target = stem.with_suffix(f".{extension}")
        fig.savefig(target, dpi=600)
        print(target)
    plt.close(fig)

    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("offered_qps", *(series[1] for series in SERIES)))
        for q in QPS:
            writer.writerow((q, *(points[q, series[0]]["actual_qps"] for series in SERIES)))
    stem.with_suffix(".txt").write_text(
        "Actual throughput versus offered QPS for UVA, Native Copy, AdaptMoE (fixed), "
        "and AdaptMoE (dynamic), using the results supplied on 2026-10-24. "
        "The source CSV preserves the full-precision values corresponding to the "
        "user's rounded table. All four curves use solid lines, distinct markers, "
        "and muted dark colors. The two-column legend is in the upper right. "
        "Straight segments connect measured points; the QPS axis is linear.\n"
        "Protocol: DeepSeek-V2-Lite-Chat BF16, 128 ShareGPT requests, "
        "1024 input / 16 output tokens, concurrency limit 128; fixed KV1 "
        "versus dynamic KV1 to KV2. Each point is a two-process mean, "
        "including queueing and final drain.\n"
        f"Source: {args.input.resolve()}\n"
        f"Reproduce: uv run {Path(__file__).resolve()}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
