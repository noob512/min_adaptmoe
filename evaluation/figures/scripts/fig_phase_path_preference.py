#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot phase-dependent Copy versus remote-UVA preference for Qwen dual Graphs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_phase_path_preference"

DEFAULT_REPORT = REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E14_CONCISE_Full.md"
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

MODEL = "Qwen3-30B-A3B FP8"
SOURCE_WORKLOADS = ("Short", "Ordinary", "Medium", "RAG", "Long")
WORKLOADS = ("Short", "Ordinary", "Medium", "Long")
CONCURRENCIES = (1, 16)
PHASES = ("Prefill", "Decode")
UVA_MECHANISM = "Native UVA 双阶段 Graph"
COPY_MECHANISM = "Native Copy 双阶段 Graph"
COPY_COLOR = "#009E73"
UVA_COLOR = "#E69F00"
REFERENCE_COLOR = "#607D8B"
EDGE_COLOR = "#333333"
CSV_COLUMNS = (
    "model",
    "workload",
    "concurrency",
    "phase",
    "metric",
    "uva_offload_gib",
    "copy_offload_gib",
    "uva_ms",
    "copy_ms",
    "ratio_uva_over_copy",
    "preferred_path",
    "preferred_speedup",
    "status",
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


def extract_between(text: str, start_heading: str, end_heading: str) -> str:
    start = text.find(start_heading)
    end = text.find(end_heading, start + len(start_heading))
    require(start >= 0 and end > start, f"section not found: {start_heading}")
    return text[start:end]


def parse_positive(cell: str, label: str) -> float:
    value = float(cell.replace(",", ""))
    require(value > 0 and math.isfinite(value), f"invalid {label}: {cell}")
    return value


def parse_qwen_table(report: str) -> pd.DataFrame:
    qwen_section = extract_between(
        report,
        "## Qwen3-30B-A3B FP8",
        "## 覆盖与失败",
    )
    header = [
        "场景",
        "C",
        "机制",
        "实际卸载 GiB",
        "TTFT (ms)",
        "TPOT (ms)",
        "Prefill E2E (ms)",
        "Decode tok/s",
        "E2E (ms)",
        "状态",
    ]
    lines = qwen_section.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if split_markdown_row(line) == header),
        -1,
    )
    require(start >= 0, "Qwen six-mechanism table not found")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    require(len(table_lines) == 62, "Qwen table does not contain 60 cells")

    records: list[dict[str, object]] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        require(len(cells) == len(header), f"malformed Qwen table row: {line}")
        row = dict(zip(header, cells, strict=True))
        if row["机制"] not in (UVA_MECHANISM, COPY_MECHANISM):
            continue
        records.append(
            {
                "workload": row["场景"],
                "concurrency": int(row["C"]),
                "mechanism": row["机制"],
                "offload_gib": parse_positive(row["实际卸载 GiB"], "offload GiB"),
                "tpot_ms": parse_positive(row["TPOT (ms)"], "TPOT"),
                "prefill_e2e_ms": parse_positive(row["Prefill E2E (ms)"], "Prefill E2E"),
                "status_text": row["状态"],
            }
        )

    raw = pd.DataFrame.from_records(records)
    require(len(raw) == 20, "expected 20 Qwen dual-Graph rows")
    require(not raw.duplicated(["workload", "concurrency", "mechanism"]).any(),
            "duplicate Qwen dual-Graph row")
    require(set(raw["workload"]) == set(SOURCE_WORKLOADS), "Qwen workload set differs")
    require(set(raw["concurrency"]) == set(CONCURRENCIES), "Qwen concurrency set differs")
    require(raw["status_text"].str.startswith("ok").all(), "unexpected retained status")
    return raw


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    raw = parse_qwen_table(report)

    indexed = raw.set_index(["workload", "concurrency", "mechanism"])
    records: list[dict[str, object]] = []
    for workload in WORKLOADS:
        for concurrency in CONCURRENCIES:
            uva = indexed.loc[(workload, concurrency, UVA_MECHANISM)]
            copy = indexed.loc[(workload, concurrency, COPY_MECHANISM)]
            is_equal_offload = math.isclose(
                float(uva["offload_gib"]),
                float(copy["offload_gib"]),
                abs_tol=0.0005,
            )
            require(is_equal_offload, f"offload mismatch for {workload}/C{concurrency}")

            for phase in PHASES:
                metric = "Prefill E2E latency" if phase == "Prefill" else "TPOT"
                uva_ms = float(uva["prefill_e2e_ms" if phase == "Prefill" else "tpot_ms"])
                copy_ms = float(copy["prefill_e2e_ms" if phase == "Prefill" else "tpot_ms"])
                ratio = uva_ms / copy_ms
                preferred_path = "Copy" if ratio > 1 else "Remote UVA"
                preferred_speedup = ratio if ratio > 1 else 1 / ratio
                records.append(
                    {
                        "model": MODEL,
                        "workload": workload,
                        "concurrency": concurrency,
                        "phase": phase,
                        "metric": metric,
                        "uva_offload_gib": float(uva["offload_gib"]),
                        "copy_offload_gib": float(copy["offload_gib"]),
                        "uva_ms": uva_ms,
                        "copy_ms": copy_ms,
                        "ratio_uva_over_copy": ratio,
                        "preferred_path": preferred_path,
                        "preferred_speedup": preferred_speedup,
                        "status": "ok",
                        "sample_nature": "single cold-process cell",
                        "source_section": "RESULTS_E0_E14_CONCISE_Full.md Qwen six-mechanism table",
                        "validity": "valid_equal-offload_dual-Graph_comparison",
                    }
                )

    data = pd.DataFrame.from_records(records, columns=CSV_COLUMNS)
    require(len(data) == 16, "expected 16 phase/configuration rows")
    require(not data.duplicated(["workload", "concurrency", "phase"]).any(),
            "duplicate plotted configuration")

    prefill = data[(data["phase"] == "Prefill") & (data["status"] == "ok")]
    decode = data[(data["phase"] == "Decode") & (data["status"] == "ok")]
    require(len(prefill) == 8 and len(decode) == 8, "expected 8 valid points per phase")
    prefill_copy = prefill[prefill["preferred_path"] == "Copy"]
    require(len(prefill_copy) == 7, "Copy does not win 7/8 valid Prefill points")
    require(
        (round(float(prefill_copy["preferred_speedup"].min()), 2),
         round(float(prefill_copy["preferred_speedup"].max()), 2)) == (1.73, 13.29),
        "Prefill Copy speedup range differs",
    )
    require((decode["preferred_path"] == "Remote UVA").all(),
            "Remote UVA does not win 8/8 Decode points")
    require(
        (round(float(decode["preferred_speedup"].min()), 2),
         round(float(decode["preferred_speedup"].max()), 2)) == (1.45, 4.26),
        "Decode UVA speedup range differs",
    )
    return data


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.9f")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == list(CSV_COLUMNS), "CSV columns differ")
    require(len(serialized) == len(data), "serialized row count differs")
    require(serialized["ratio_uva_over_copy"].notna().sum() == 16,
            "serialized valid-ratio count differs")
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
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
    rows = [(workload, concurrency) for workload in WORKLOADS for concurrency in CONCURRENCIES]
    y_positions = np.array(
        [group * 1.45 + within * 0.62 for group in range(len(WORKLOADS)) for within in range(2)]
    )
    y_labels = [f"{workload} / C{concurrency}" for workload, concurrency in rows]

    for panel_index, (axis, phase) in enumerate(zip(axes, PHASES, strict=True)):
        panel = data[data["phase"] == phase].set_index(["workload", "concurrency"]).loc[rows]
        axis.set_xscale("log", base=2)
        axis.set_xlim(0.18, 18)
        axis.set_ylim(y_positions[-1] + 0.55, -0.95)
        axis.axvline(1, color=REFERENCE_COLOR, linewidth=1.15, zorder=1)

        for y, ((workload, concurrency), row) in zip(
            y_positions, panel.iterrows(), strict=True
        ):
            axis.hlines(y, 0.18, 18, color="#E7E7E7", linewidth=0.55, zorder=0)

            ratio = float(row["ratio_uva_over_copy"])
            speedup = float(row["preferred_speedup"])
            preferred = str(row["preferred_path"])
            color = COPY_COLOR if preferred == "Copy" else UVA_COLOR
            marker = "s" if preferred == "Copy" else "^"
            axis.hlines(y, min(1, ratio), max(1, ratio), color="#AFAFAF", linewidth=1.0, zorder=2)
            axis.scatter(
                [ratio],
                [y],
                marker=marker,
                s=32,
                color=color,
                edgecolor=EDGE_COLOR,
                linewidth=0.65,
                zorder=4,
            )
            if ratio > 1:
                label_x, horizontal_alignment = ratio * 1.08, "left"
            elif ratio < 0.35:
                label_x, horizontal_alignment = ratio * 1.10, "left"
            else:
                label_x, horizontal_alignment = ratio / 1.08, "right"
            axis.text(
                label_x,
                y,
                f"{speedup:.2f}×",
                ha=horizontal_alignment,
                va="center",
                fontsize=7.0,
                color="#222222",
            )

        axis.set_yticks(y_positions, y_labels)
        axis.set_title(f"({chr(ord('a') + panel_index)}) {phase}", loc="left", pad=4)
        axis.text(
            0.02,
            0.97,
            "Remote UVA preferred",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color=UVA_COLOR,
        )
        axis.text(
            0.98,
            0.97,
            "Copy preferred",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color=COPY_COLOR,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=5)
        axis.tick_params(axis="x", length=3, width=0.7)
        axis.xaxis.set_minor_formatter(NullFormatter())

    ticks = (0.25, 0.5, 1, 2, 4, 8, 16)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(("4×", "2×", "1×", "2×", "4×", "8×", "16×"))
    axes[-1].set_xlim(0.18, 18)
    axes[-1].set_xlabel("Preferred-path speedup (log scale)")

    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="s", linestyle="none", markersize=5.0,
                markerfacecolor=COPY_COLOR, markeredgecolor=EDGE_COLOR,
                label="Copy lower latency",
            ),
            Line2D(
                [0], [0], marker="^", linestyle="none", markersize=5.2,
                markerfacecolor=UVA_COLOR, markeredgecolor=EDGE_COLOR,
                label="Remote UVA lower latency",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.002),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.18, right=0.965, bottom=0.095, top=0.925, hspace=0.27)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, format="pdf")
    figure.savefig(png_path, format="png", dpi=300)
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "Phase-dependent data-path preference for Qwen3-30B-A3B FP8 under equal-offload "
        "dual-Graph execution at concurrency 1 and 16, excluding the RAG workload. Each "
        "point compares Native UVA "
        "with Native Copy for the same workload and concurrency; positions right of 1× "
        "favor Copy and positions left of 1× favor remote UVA. Prefill uses batch Prefill "
        "E2E latency, for which Copy is faster in 7/8 configurations by 1.73–13.29×. "
        "Decode uses request TPOT, for which remote UVA is faster in 8/8 configurations "
        "by 1.45–4.26× using the displayed source values. Each "
        "configuration is a single cold-process sample, so no error bars are shown."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_summary(data: pd.DataFrame) -> None:
    for phase in PHASES:
        panel = data[(data["phase"] == phase) & (data["status"] == "ok")]
        counts = panel["preferred_path"].value_counts().to_dict()
        print(f"{phase}: {len(panel)} valid; preference counts={counts}")
        for preferred_path in ("Copy", "Remote UVA"):
            subset = panel[panel["preferred_path"] == preferred_path]
            if not subset.empty:
                print(
                    f"  {preferred_path}: {subset['preferred_speedup'].min():.6f}–"
                    f"{subset['preferred_speedup'].max():.6f}x"
                )
    print("Excluded workload: RAG (C1 and C16)")


def main() -> None:
    args = parse_args()
    data = build_data(args.report)
    serialized = write_and_reload_csv(data, args.csv)
    render_figure(serialized, args.pdf, args.png)
    write_caption(args.caption)
    print_summary(serialized)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png} (300 dpi)")
    print(f"Wrote {args.caption}")


if __name__ == "__main__":
    main()
