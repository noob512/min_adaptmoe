#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot the measured E13 K2/K4/K6/Adaptive throughput crossover."""

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
FIGURE_NAME = "fig_e13_rag_pressure_throughput"

DEFAULT_REPORT = (
    REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md"
)
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

CONCURRENCIES = tuple(range(1, 8))
PRESSURE_BY_CONCURRENCY = {
    1: "Low",
    2: "Low",
    3: "Medium",
    4: "Medium",
    5: "Medium",
    6: "High",
    7: "High",
}
MECHANISM_MAP = {
    "Fixed-K2": "Fixed-K2",
    "Fixed-K4": "Fixed-K4",
    "Fixed-K6": "Fixed-K6",
    "Adaptive": "AdaptMoE",
}
MECHANISMS = ("Fixed-K2", "Fixed-K4", "Fixed-K6", "AdaptMoE")
DISPLAY_LABELS = {
    "Fixed-K2": "Fixed-low",
    "Fixed-K4": "Fixed-mid",
    "Fixed-K6": "Fixed-high",
    "AdaptMoE": "AdaptMoE",
}
KV_TARGETS = {
    "Fixed-K2": 1364,
    "Fixed-K4": 2730,
    "Fixed-K6": 4095,
}
ADAPTIVE_TARGETS = {
    1: 1364,
    2: 1364,
    3: 2730,
    4: 2730,
    5: 2730,
    6: 4095,
    7: 4095,
}
STYLES = {
    "AdaptMoE": {
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.5,
        "markersize": 3.8,
    },
    "Fixed-K2": {
        "color": "#E69F00",
        "linestyle": "-",
        "linewidth": 1.5,
        "markersize": 5.0,
    },
    "Fixed-K4": {
        "color": "#009E73",
        "linestyle": "-",
        "linewidth": 1.5,
        "markersize": 5.0,
    },
    "Fixed-K6": {
        "color": "#7E57C2",
        "linestyle": "-",
        "linewidth": 1.5,
        "markersize": 5.0,
    },
}
CSV_COLUMNS = (
    "model",
    "workload",
    "pressure",
    "concurrency",
    "source_mechanism",
    "mechanism",
    "kv_target_blocks",
    "output_tokens_per_s",
    "max_concurrent_requests",
    "requested_concurrent_requests",
    "best_fixed_mechanism",
    "ratio_to_best_fixed",
    "target_fixed_mechanism",
    "adaptive_target_fixed_ratio",
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


def extract_section(report: str) -> str:
    start_heading = "### 16.6 RAG-8192 并发 B1--B8 四臂 sweep"
    end_heading = "## 17. E14：同请求集冷专家与固定整层卸载的 eager UVA 扩展性"
    start = report.find(start_heading)
    end = report.find(end_heading, start + len(start_heading))
    require(start >= 0 and end > start, "E13 §16.6 not found")
    return report[start:end]


def parse_positive(cell: str, label: str) -> float:
    cleaned = cell.strip().replace(",", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?: ms)?", cleaned)
    require(match is not None, f"invalid {label}: {cell}")
    value = float(match.group(1))
    require(value > 0 and math.isfinite(value), f"non-positive {label}: {cell}")
    return value


def parse_ratio(cell: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)×", cell.strip())
    require(match is not None, f"invalid ratio: {cell}")
    value = float(match.group(1))
    require(value > 0 and math.isfinite(value), f"invalid ratio: {cell}")
    return value


def parse_active_requests(cell: str, concurrency: int) -> dict[str, int]:
    match = re.fullmatch(r"(\d+)/(\d+)/(\d+)/(\d+)", cell.strip())
    require(match is not None, f"invalid concurrent-request counts: {cell}")
    values = [int(match.group(index)) for index in range(1, 5)]
    require(all(0 < value <= concurrency for value in values),
            f"invalid concurrent-request counts: {cell}")
    return dict(zip(MECHANISMS, values, strict=True))


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    section = extract_section(report)
    require("B1--2使用K2、B3--5使用K4、B6--8使用K6" in section,
            "Adaptive target-tier schedule differs")
    require("每个cell先运行一次排除\nwarmup，再取两次正式进程内样本的中位数" in section,
            "sample nature differs")
    require("verifier为`status=ok`且101/101项检查通过" in section,
            "verification status differs")

    header = [
        "B", "Fixed-K2 output tok/s", "Fixed-K4", "Fixed-K6", "Adaptive",
        "最佳Fixed", "Adaptive/目标Fixed", "最大同时运行 K2/K4/K6/A",
    ]
    lines = section.splitlines()
    start = next((i for i, line in enumerate(lines) if split_markdown_row(line) == header), -1)
    require(start >= 0, "§16.6 table header not found")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    require(len(table_lines) == 10, "§16.6 table row count differs")

    records: list[dict[str, object]] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        require(len(cells) == len(header), f"malformed §16.6 row: {line}")
        row = dict(zip(header, cells, strict=True))
        concurrency = int(row["B"])
        if concurrency == 8:
            continue
        require(concurrency in CONCURRENCIES, f"unexpected concurrency: B{concurrency}")
        pressure = PRESSURE_BY_CONCURRENCY[concurrency]
        active_requests = parse_active_requests(
            row["最大同时运行 K2/K4/K6/A"], concurrency
        )
        best_fixed = f"Fixed-{row['最佳Fixed']}"
        require(best_fixed in KV_TARGETS, f"unexpected best Fixed tier: {row['最佳Fixed']}")
        target_fixed = (
            "Fixed-K2" if concurrency <= 2 else "Fixed-K4" if concurrency <= 5 else "Fixed-K6"
        )
        target_ratio = parse_ratio(row["Adaptive/目标Fixed"])
        source_values = {
            "Fixed-K2": row["Fixed-K2 output tok/s"],
            "Fixed-K4": row["Fixed-K4"],
            "Fixed-K6": row["Fixed-K6"],
            "Adaptive": row["Adaptive"],
        }
        for source_mechanism, value_cell in source_values.items():
            mechanism = MECHANISM_MAP[source_mechanism]
            kv_target = (
                ADAPTIVE_TARGETS[concurrency]
                if mechanism == "AdaptMoE"
                else KV_TARGETS[mechanism]
            )
            records.append(
                {
                    "model": "Qwen3-30B-A3B FP8",
                    "workload": "RAG-8192 → 16",
                    "pressure": pressure,
                    "concurrency": concurrency,
                    "source_mechanism": source_mechanism,
                    "mechanism": mechanism,
                    "kv_target_blocks": kv_target,
                    "output_tokens_per_s": parse_positive(value_cell, "output tok/s"),
                    "max_concurrent_requests": active_requests[mechanism],
                    "requested_concurrent_requests": concurrency,
                    "best_fixed_mechanism": best_fixed,
                    "target_fixed_mechanism": target_fixed,
                    "adaptive_target_fixed_ratio": target_ratio,
                    "status": "ok",
                    "sample_nature": (
                        "median of two within-process measurements in one cold-process arm"
                    ),
                    "source_section": (
                        "RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md "
                        "§16.6 RAG-8192 concurrency B1--B8 four-arm sweep"
                    ),
                    "validity": "valid_hash_matched_single_cold_process_arm",
                }
            )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 28, "expected 28 measured throughput points after excluding B8")
    require(not data.duplicated(["concurrency", "mechanism"]).any(), "duplicate point")
    require(set(data["mechanism"]) == set(MECHANISMS), "mechanism coverage differs")

    data["ratio_to_best_fixed"] = math.nan
    for concurrency in CONCURRENCIES:
        mask = data["concurrency"] == concurrency
        fixed = data[mask & data["mechanism"].str.startswith("Fixed-")]
        best_row = fixed.loc[fixed["output_tokens_per_s"].idxmax()]
        best_mechanism = str(best_row["mechanism"])
        best_value = float(best_row["output_tokens_per_s"])
        require(best_mechanism == str(data.loc[mask, "best_fixed_mechanism"].iloc[0]),
                f"best Fixed tier differs at B{concurrency}")
        data.loc[mask, "ratio_to_best_fixed"] = data.loc[mask, "output_tokens_per_s"] / best_value
        adaptive = data.loc[mask & (data["mechanism"] == "AdaptMoE")].iloc[0]
        target_value = float(
            data.loc[mask & (data["mechanism"] == adaptive["target_fixed_mechanism"]),
                     "output_tokens_per_s"].iloc[0]
        )
        computed_target_ratio = float(adaptive["output_tokens_per_s"]) / target_value
        require(round(computed_target_ratio, 3) == float(adaptive["adaptive_target_fixed_ratio"]),
                f"Adaptive/target-Fixed ratio differs at B{concurrency}")

    mechanism_rank = {mechanism: i for i, mechanism in enumerate(MECHANISMS)}
    data["_mechanism"] = data["mechanism"].map(mechanism_rank)
    data = data.sort_values(["concurrency", "_mechanism"]).drop(columns="_mechanism")
    return data.loc[:, CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.6f")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == list(CSV_COLUMNS), "CSV columns differ")
    require(len(serialized) == 28, "serialized point count differs")
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8.4,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.35, 2.7))
    plot_order = ("Fixed-K2", "Fixed-K4", "Fixed-K6", "AdaptMoE")
    for mechanism in plot_order:
        series = data[data["mechanism"] == mechanism].sort_values("concurrency")
        require(len(series) == len(CONCURRENCIES), f"series coverage differs: {mechanism}")
        style = STYLES[mechanism]
        axis.plot(
            series["concurrency"],
            series["output_tokens_per_s"],
            color=style["color"],
            marker="o",
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markerfacecolor=style["color"] if mechanism == "AdaptMoE" else "white",
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            label=DISPLAY_LABELS[mechanism],
            zorder=5 if mechanism == "AdaptMoE" else 3,
        )

    axis.set_xlim(0.8, 7.2)
    axis.set_ylim(8, 17.2)
    axis.set_xticks(CONCURRENCIES)
    axis.yaxis.set_major_locator(MultipleLocator(2))
    axis.set_xlabel("Concurrency (B)")
    axis.set_ylabel("Output throughput (tokens/s)")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DCDCDC", linewidth=0.6)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0], [0],
            color=STYLES[mechanism]["color"],
            marker="o",
            linestyle=STYLES[mechanism]["linestyle"],
            linewidth=STYLES[mechanism]["linewidth"],
            markersize=STYLES[mechanism]["markersize"],
            markerfacecolor=STYLES[mechanism]["color"] if mechanism == "AdaptMoE" else "white",
            markeredgecolor=STYLES[mechanism]["color"],
            markeredgewidth=0.9,
            label=DISPLAY_LABELS[mechanism],
        )
        for mechanism in ("AdaptMoE", "Fixed-K2", "Fixed-K4", "Fixed-K6")
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.15,
    )
    figure.subplots_adjust(left=0.19, right=0.975, bottom=0.20, top=0.78)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Throughput crossover across KV-pressure tiers",
        "Subject": "Fixed-K2, Fixed-K4, Fixed-K6, and AdaptMoE on a common RAG trace",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(png_path, format="png", dpi=300, metadata={"Software": SCRIPT_PATH.name})
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "Measured output throughput on Qwen3-30B-A3B FP8 with the same RAG-8192→16 "
        "request under eager prefill and remote decode across concurrency B1--B7. The best "
        "fixed KV tier is K2 at B1--B2, K4 at B3--B5, and K6 at B6--B7. "
        "Initialization, tier-change transactions, and one target-tier warmup are excluded. "
        "Each arm is one "
        "cold process and each point is the median of two within-process measurements; no "
        "error bars or statistical-equivalence claim is made. The plotted cells produced "
        "matching output hashes."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_summary(data: pd.DataFrame) -> None:
    for concurrency in CONCURRENCIES:
        pressure = PRESSURE_BY_CONCURRENCY[concurrency]
        panel = data[data["concurrency"] == concurrency]
        ordered = panel.sort_values("output_tokens_per_s", ascending=False)
        ranking = " > ".join(
            f"{row.mechanism} ({row.output_tokens_per_s:.4f})" for row in ordered.itertuples()
        )
        adaptive_ratio = float(
            panel.loc[panel["mechanism"] == "AdaptMoE", "ratio_to_best_fixed"].iloc[0]
        )
        print(f"[SUMMARY] {pressure}/B{concurrency}: {ranking}; AdaptMoE/best Fixed={adaptive_ratio:.4f}×")


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    require(report_path.is_file(), f"report not found: {report_path}")
    data = build_data(report_path)
    plot_data = write_and_reload_csv(data, args.csv.resolve())
    render_figure(plot_data, args.pdf.resolve(), args.png.resolve())
    write_caption(args.caption.resolve())
    print(f"[OK] sole source: {report_path}")
    print(f"[OK] CSV: {args.csv.resolve()} ({len(plot_data)} rows)")
    print(f"[OK] PDF: {args.pdf.resolve()}")
    print(f"[OK] PNG: {args.png.resolve()}")
    print(f"[OK] caption: {args.caption.resolve()}")
    print_summary(plot_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
