#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot measured KV/expert demand and the fixed-partition crossover."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
FIGURES = SCRIPT.parents[1]
NAME = "fig_kv_expert_partition_crossover"
DEFAULT_VERIFICATION = REPO_ROOT / (
    "experiments/eurosys_moe_offload_evaluation/artifacts/"
    "e13-partition-motivation-20260915-attempt-02/verification.json"
)

WORKLOAD_LABELS = {"long_c2": "Long-C2", "short_b128": "Short-B128"}
PLAN_LABELS = {
    "fixed_k2": "Expert-rich / 2-GiB KV",
    "fixed_k4": "Balanced / 4-GiB KV",
    "fixed_k6": "KV-rich / 6-GiB KV",
}
PLAN_COLORS = {
    "fixed_k2": "#E69F00",
    "fixed_k4": "#009E73",
    "fixed_k6": "#8E63B6",
}
PLAN_HATCHES = {"fixed_k2": "//", "fixed_k4": "..", "fixed_k6": "xx"}
KV_COLOR = "#8E63B6"
EXPERT_COLOR = "#0072B2"
EDGE = "#333333"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--csv", type=Path, default=FIGURES / "data" / f"{NAME}.csv")
    parser.add_argument("--pdf", type=Path, default=FIGURES / "pdf" / f"{NAME}.pdf")
    parser.add_argument("--png", type=Path, default=FIGURES / "png" / f"{NAME}.png")
    parser.add_argument(
        "--caption", type=Path, default=FIGURES / "captions" / f"{NAME}.txt"
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_data(path: Path) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("status") == "ok", "verification is not valid")
    rows = pd.DataFrame(payload["rows"])
    diagnostics = pd.DataFrame(payload["diagnostics"])
    require(len(rows) == 12 and len(diagnostics) == 2, "unexpected process count")
    require(rows["repeat_index"].nunique() == 2, "performance repeats differ")

    valid = rows[rows["status"] == "ok"].copy()
    medians = (
        valid.groupby(["workload", "arm"], as_index=False)["output_tokens_per_s"]
        .median()
        .rename(columns={"output_tokens_per_s": "median_output_tokens_per_s"})
    )
    rows = rows.merge(medians, on=["workload", "arm"], how="left")
    best = (
        medians.groupby("workload", as_index=False)["median_output_tokens_per_s"]
        .max()
        .rename(columns={"median_output_tokens_per_s": "best_output_tokens_per_s"})
    )
    rows = rows.merge(best, on="workload", how="left")
    rows["normalized_output_throughput"] = np.where(
        rows["status"].eq("ok"),
        rows["output_tokens_per_s"] / rows["best_output_tokens_per_s"],
        np.nan,
    )
    rows["workload_label"] = rows["workload"].map(WORKLOAD_LABELS)
    rows["plan_label"] = rows["arm"].map(PLAN_LABELS)
    rows["source_section"] = rows["source_result"].map(
        lambda value: f"{path}: {value}"
    )

    diagnostics["workload_label"] = diagnostics["workload"].map(WORKLOAD_LABELS)
    diagnostics["plan_label"] = diagnostics["arm"].map(PLAN_LABELS)
    diagnostics["source_section"] = diagnostics["source_result"].map(
        lambda value: f"{path}: {value}"
    )
    combined = pd.concat([rows, diagnostics], ignore_index=True, sort=False)
    require(not combined["workload_label"].isna().any(), "workload mapping failed")
    return payload, combined


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.3,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "hatch.linewidth": 0.55,
        }
    )


def median_row(data: pd.DataFrame, workload: str, arm: str) -> pd.Series:
    values = data[
        (data["kind"] == "performance")
        & (data["workload"] == workload)
        & (data["arm"] == arm)
    ]
    require(len(values) == 2, f"missing performance cell {workload}/{arm}")
    return values.iloc[0]


def render(payload: dict, data: pd.DataFrame, pdf: Path, png: Path) -> None:
    configure()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.62), gridspec_kw={"wspace": 0.42})
    workloads = ("long_c2", "short_b128")
    labels = [WORKLOAD_LABELS[item] for item in workloads]
    x = np.arange(2)

    # (a) Exact block demand, confirmed by the fully resident plan.
    ax = axes[0]
    kv_values = [float(payload["claims"][f"{item}_requires_gib"]) for item in workloads]
    bars = ax.bar(
        x,
        kv_values,
        width=0.52,
        color=KV_COLOR,
        edgecolor=EDGE,
        linewidth=0.7,
        hatch="//",
        zorder=3,
    )
    for index, (bar, value) in enumerate(zip(bars, kv_values, strict=True)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.13,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    for arm, limit in zip(("fixed_k2", "fixed_k4", "fixed_k6"), (2, 4, 6), strict=True):
        ax.axhline(
            limit,
            color=PLAN_COLORS[arm],
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        ax.text(
            1.48,
            limit + 0.06,
            f"{limit} GiB",
            color=PLAN_COLORS[arm],
            ha="right",
            va="bottom",
            fontsize=6.7,
        )
    ax.set_ylim(0, 6.65)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Required GPU KV capacity (GiB)")
    ax.set_title("(a) KV capacity pressure", loc="left", pad=3)

    # (b) Diagnostic routing window; counter overhead is excluded from performance.
    ax = axes[1]
    diagnostic_x = np.array([0.25, 0.75])
    diag = data[data["kind"] == "diagnostic"].set_index("workload").loc[list(workloads)]
    unique = diag["unique_active_experts_per_layer_step"].astype(float).to_numpy()
    ax.vlines(diagnostic_x, 0, unique, color="#AEBFCC", linewidth=1.1, zorder=1)
    ax.scatter(
        diagnostic_x,
        unique,
        s=35,
        marker="o",
        facecolor=EXPERT_COLOR,
        edgecolor=EDGE,
        linewidth=0.7,
        zorder=3,
    )
    for index, (xpos, value) in enumerate(zip(diagnostic_x, unique, strict=True)):
        ax.annotate(
            f"{value:.1f}",
            xy=(xpos, value),
            xytext=((2 if index == 0 else -2), 7),
            textcoords="offset points",
            ha="left" if index == 0 else "right",
            va="bottom",
            fontsize=6.7,
            linespacing=1.05,
        )
    ax.set_ylim(0, max(unique) * 1.34)
    ax.set_xlim(0, 1)
    ax.set_xticks(diagnostic_x, labels)
    ax.set_ylabel("Unique active experts\nper layer per decode step")
    ax.set_title("(b) Expert working set", loc="left", pad=3)

    # (c) Median across two cold processes, normalized within workload.
    ax = axes[2]
    width = 0.23
    offsets = {"fixed_k2": -width, "fixed_k4": 0.0, "fixed_k6": width}
    for arm in ("fixed_k2", "fixed_k4", "fixed_k6"):
        for workload_index, workload in enumerate(workloads):
            cell = data[
                (data["kind"] == "performance")
                & (data["workload"] == workload)
                & (data["arm"] == arm)
            ]
            status = str(cell.iloc[0]["status"])
            xpos = workload_index + offsets[arm]
            if status != "ok":
                continue
            median_tps = float(cell["median_output_tokens_per_s"].iloc[0])
            best_tps = float(cell["best_output_tokens_per_s"].iloc[0])
            height = median_tps / best_tps
            ax.bar(
                [xpos],
                [height],
                width=width * 0.88,
                color=PLAN_COLORS[arm],
                edgecolor=EDGE,
                linewidth=0.65,
                hatch=PLAN_HATCHES[arm],
                zorder=2,
            )
            raw_points = cell["output_tokens_per_s"].astype(float).to_numpy() / best_tps
            jitter = np.array([-0.035, 0.035])
            ax.scatter(
                xpos + jitter,
                raw_points,
                s=13,
                marker="o",
                facecolor="white",
                edgecolor=EDGE,
                linewidth=0.7,
                zorder=4,
            )
            ax.text(
                xpos,
                height
                + (
                    {"fixed_k2": 0.095, "fixed_k4": 0.060, "fixed_k6": 0.025}[arm]
                    if workload == "short_b128"
                    else 0.045
                ),
                f"{median_tps:.1f}" if median_tps < 100 else f"{median_tps:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    ax.set_ylim(0, 1.23)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalized output throughput")
    ax.set_title("(c) Fixed-partition crossover", loc="left", pad=3)

    for ax in axes:
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.55, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=4)

    fig.legend(
        handles=[
            Patch(
                facecolor=PLAN_COLORS[arm],
                edgecolor=EDGE,
                hatch=PLAN_HATCHES[arm],
                linewidth=0.65,
                label=PLAN_LABELS[arm],
            )
            for arm in ("fixed_k2", "fixed_k4", "fixed_k6")
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.82, wspace=0.43)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)


def write_caption(path: Path) -> None:
    caption = (
        "KV–expert partition demand on Qwen3-30B-A3B FP8 under a fixed HBM budget. "
        "(a) Long-C2 uses two 26,000-token prompts and requires 4.78 GiB of GPU KV "
        "capacity, whereas Short-B128 with 128-token prompts and 32-token outputs "
        "requires 1.88 GiB. (b) Separate instrumented cold processes report the "
        "routing working set over 16 steady Decode steps. (c) Throughput is normalized "
        "to the best fully admitted fixed "
        "partition within each workload. Long-C2 results for the 2- and 4-GiB KV "
        "partitions are omitted because they lack sufficient simultaneous KV capacity; "
        "they are not encoded as zero. Output throughput counts all generated "
        "tokens over full request wall time, including Prefill and Decode. Bars show "
        "medians and white points show two independent cold processes; routing counters "
        "were disabled in all performance processes. The result is limited to the "
        "evaluated dynamic workloads and eager execution."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload, data = load_data(args.verification)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.csv, index=False, float_format="%.9f")
    render(payload, data, args.pdf, args.png)
    write_caption(args.caption)
    short = payload["median_output_tokens_per_s"]["short_b128"]
    print(
        "Short-B128 median output tok/s: "
        + ", ".join(f"{arm}={float(value):.4f}" for arm, value in short.items())
    )
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png} (300 dpi)")
    print(f"Wrote {args.caption}")


if __name__ == "__main__":
    main()
