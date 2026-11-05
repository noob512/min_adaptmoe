#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot AdaptMoE throughput across three measured ShareGPT pressure levels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
FIGURES = SCRIPT.parents[1]
NAME = "fig_adaptmoe_pressure_throughput"

INPUTS = {
    "128 × 8": REPO_ROOT
    / "experiments/adaptmoe_token_threshold_sharegpt_compare/artifacts/"
    "park16-audit-off-p128-c8-20260915-attempt-01/verification.json",
    "512 × 16": REPO_ROOT
    / "experiments/adaptmoe_token_threshold_sharegpt_compare/artifacts/"
    "park16-audit-off-p512-c16-20260915-attempt-01/verification.json",
    "512 × 32": REPO_ROOT
    / "experiments/adaptmoe_token_threshold_sharegpt_compare/artifacts/"
    "park16-audit-off-p512-c32-20260915-attempt-01/verification.json",
}

MECHANISMS = ("adaptive", "native_uva", "native_copy")
LABELS = {
    "adaptive": "AdaptMoE",
    "native_uva": "Native UVA",
    "native_copy": "Native Copy",
}
COLORS = {
    "adaptive": "#0072B2",
    "native_uva": "#E69F00",
    "native_copy": "#009E73",
}
HATCHES = {"adaptive": "//", "native_uva": "..", "native_copy": "\\\\"}
EDGE = "#333333"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for workload_index, (workload, path) in enumerate(INPUTS.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        require(payload.get("status") == "ok", f"invalid verification: {path}")
        require(
            payload.get("strict_cross_arm_output_equivalence") is True,
            f"cross-arm outputs differ: {path}",
        )
        require(int(protocol["park_token_threshold"]) == 16, "PARK threshold differs")
        require(int(protocol["resume_token_threshold"]) == 128, "RESUME threshold differs")
        require(int(protocol["samples_per_arm"]) == 2, "sample count differs")
        require(protocol["cudagraph_mode"] == "NONE", "expected eager execution")
        cases = payload["cases"]
        require(len(cases) == 6, f"expected six cold processes: {path}")
        for mechanism in MECHANISMS:
            selected = [case for case in cases if case["arm"] == mechanism]
            require(len(selected) == 2, f"missing repeats for {workload}/{mechanism}")
            require(all(not case["errors"] for case in selected), "case errors present")
            measured_mean = float(
                np.mean([case["request_throughput_per_second"] for case in selected])
            )
            reported_mean = float(
                payload["aggregate"][mechanism]["request_throughput_per_second"]
            )
            require(
                abs(measured_mean - reported_mean) < 1e-9,
                f"aggregate mismatch for {workload}/{mechanism}",
            )
            for case in selected:
                rows.append(
                    {
                        "model": "DeepSeek-V2-Lite-Chat BF16",
                        "workload": workload,
                        "workload_order": workload_index,
                        "prompt_tokens": int(protocol["prompt_tokens_per_request"]),
                        "concurrency": int(protocol["requests"]),
                        "output_tokens_per_request": int(
                            protocol["output_tokens_per_request"]
                        ),
                        "mechanism": LABELS[mechanism],
                        "mechanism_key": mechanism,
                        "sample": int(case["sample"]),
                        "execution_ordinal": int(case["ordinal"]),
                        "request_throughput_req_s": float(
                            case["request_throughput_per_second"]
                        ),
                        "ttft_ms": float(case["mean_ttft_ms"]),
                        "tpot_ms": float(case["mean_tpot_ms"]),
                        "e2e_ms": float(case["mean_e2e_ms"]),
                        "offloaded_bytes": int(case["offloaded_bytes"]),
                        "offloaded_gib": int(case["offloaded_bytes"]) / 2**30,
                        "measured_copy_bytes": int(case["measured_copy_bytes"]),
                        "status": "ok",
                        "source_section": f"{path}: cases/{case['directory']}",
                        "validity": "valid_independent_cold_process_no_errorbar",
                    }
                )

    data = pd.DataFrame(rows)
    means = (
        data.groupby(["workload", "mechanism_key"], as_index=False)[
            "request_throughput_req_s"
        ]
        .mean()
        .rename(columns={"request_throughput_req_s": "mean_request_throughput_req_s"})
    )
    data = data.merge(means, on=["workload", "mechanism_key"], how="left")
    pivot = means.pivot(
        index="workload", columns="mechanism_key", values="mean_request_throughput_req_s"
    )
    ratios = pd.DataFrame(
        {
            "workload": pivot.index.to_numpy(),
            "adaptive_over_uva": (
                pivot["adaptive"] / pivot["native_uva"]
            ).to_numpy(),
            "adaptive_over_copy": (
                pivot["adaptive"] / pivot["native_copy"]
            ).to_numpy(),
        }
    )
    data = data.merge(ratios, on="workload", how="left")
    require(len(data) == 18, "unexpected final row count")
    return data.sort_values(
        ["workload_order", "mechanism_key", "sample"], ignore_index=True
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.8,
            "hatch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def render(data: pd.DataFrame, pdf: Path, png: Path) -> None:
    configure_style()
    workloads = list(INPUTS)
    x = np.arange(len(workloads), dtype=float)
    fig, ax = plt.subplots(figsize=(3.35, 2.60))

    width = 0.235
    offsets = {"adaptive": -width, "native_uva": 0.0, "native_copy": width}
    for mechanism in MECHANISMS:
        means: list[float] = []
        for workload in workloads:
            cell = data[
                (data["workload"] == workload)
                & (data["mechanism_key"] == mechanism)
            ]
            means.append(float(cell["mean_request_throughput_req_s"].iloc[0]))
        xpos = x + offsets[mechanism]
        bars = ax.bar(
            xpos,
            means,
            width=width * 0.91,
            color=COLORS[mechanism],
            edgecolor=EDGE,
            linewidth=1.0 if mechanism == "adaptive" else 0.65,
            hatch=HATCHES[mechanism],
            zorder=2,
            label=LABELS[mechanism],
        )
        for workload_index, (bar, mean) in enumerate(zip(bars, means, strict=True)):
            cell = data[
                (data["workload"] == workloads[workload_index])
                & (data["mechanism_key"] == mechanism)
            ]
            points = cell["request_throughput_req_s"].to_numpy(dtype=float)
            ax.scatter(
                xpos[workload_index] + np.array([-0.027, 0.027]),
                points,
                s=14,
                marker="o",
                facecolor="white",
                edgecolor=EDGE,
                linewidth=0.65,
                zorder=4,
            )
            if mechanism == "adaptive":
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + 0.31,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    color=COLORS[mechanism],
                )

    ax.set_ylim(0, 12.35)
    ax.set_xticks(x, workloads)
    ax.set_xlabel("Prompt length × concurrency")
    ax.set_ylabel("Request throughput (req/s)")
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)

    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS[mechanism],
                edgecolor=EDGE,
                linewidth=1.0 if mechanism == "adaptive" else 0.65,
                hatch=HATCHES[mechanism],
                label=LABELS[mechanism],
            )
            for mechanism in MECHANISMS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=0.65,
        handlelength=1.15,
    )
    fig.subplots_adjust(left=0.17, right=0.995, bottom=0.22, top=0.81)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)


def write_caption(path: Path) -> None:
    caption = (
        "Request throughput of AdaptMoE, Native UVA, and Native Copy on "
        "DeepSeek-V2-Lite-Chat BF16 under three frozen ShareGPT workloads. PARK and "
        "RESUME thresholds are 16 and 128 tokens, respectively; execution is eager "
        "without CUDA Graphs, and high-overhead residency diagnostics are disabled. "
        "Bars show means and white points show two independent cold processes. "
        "Native UVA is 6.33% faster at "
        "128 × 8, AdaptMoE is 1.95× faster than UVA and 1.27× faster than Copy at "
        "512 × 16, and AdaptMoE and Copy differ by only 0.35% at 512 × 32. All 18 "
        "processes completed with matching cross-arm output hashes. No error bars are "
        "shown because each cell has only two independent samples. AdaptMoE offloads "
        "3.13% more parameter bytes than the native arms, so this is not a strict "
        "equal-offload microbenchmark and the result is limited to these workloads."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    data = load_rows()
    csv_path = FIGURES / "data" / f"{NAME}.csv"
    pdf_path = FIGURES / "pdf" / f"{NAME}.pdf"
    png_path = FIGURES / "png" / f"{NAME}.png"
    caption_path = FIGURES / "captions" / f"{NAME}.txt"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False, float_format="%.9f")
    render(data, pdf_path, png_path)
    write_caption(caption_path)
    for workload in INPUTS:
        row = data[data["workload"] == workload].iloc[0]
        print(
            f"{workload}: AdaptMoE/UVA={row['adaptive_over_uva']:.4f}, "
            f"AdaptMoE/Copy={row['adaptive_over_copy']:.4f}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path} (300 dpi)")
    print(f"Wrote {caption_path}")


if __name__ == "__main__":
    main()
