#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Generate the Stock UVM demand-paging figure from REPORT.md §11.3."""

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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_REPORT = REPO_ROOT / "evaluation/data/processed/REPORT.md"
FIGURE_NAME = "fig_stock_uvm_demand_paging"
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

SECTION_START = "### 11.3 vLLM Device-resident 与 Stock UVM Demand 单轮对照"
SECTION_END = "### 11.4 H 的结论和限制"

MODEL_BY_SUBSECTION = {
    "11.3.1": "DeepSeek-V2-Lite BF16",
    "11.3.2": "Qwen3-30B-A3B FP8",
}
EXPECTED_CONFIGS = {
    "DeepSeek-V2-Lite BF16": [
        ("Short", 1),
        ("Short", 16),
        ("Ordinary", 1),
        ("Long", 1),
    ],
    "Qwen3-30B-A3B FP8": [
        ("Short", 1),
        ("Short", 16),
        ("Ordinary", 1),
        ("Medium", 1),
        ("RAG", 1),
        ("Long", 1),
    ],
}
METRICS = ("TTFT", "TPOT", "E2E")
TTFT_ANNOTATION_BY_MODEL = {
    # Publication labels reported in §11.3 and explicitly requested for the figure.
    "DeepSeek-V2-Lite BF16": 25.88,
    "Qwen3-30B-A3B FP8": 24.80,
}

# Okabe–Ito colors; markers follow the requested metric encoding.
METRIC_STYLE = {
    "TTFT": {"color": "#0072B2", "marker": "o", "offset": -0.22},
    "TPOT": {"color": "#E69F00", "marker": "s", "offset": 0.00},
    "E2E": {"color": "#009E73", "marker": "^", "offset": 0.22},
}


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


def parse_number(cell: str) -> float | None:
    value = cell.strip()
    if value == "OOM":
        return None
    return float(value.replace(",", ""))


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def extract_section(report_path: Path) -> str:
    report = report_path.read_text(encoding="utf-8")
    start = report.find(SECTION_START)
    end = report.find(SECTION_END, start)
    require(start >= 0 and end > start, "REPORT.md §11.3 boundaries not found")
    section = report[start:end]
    require("64 MiB Stock UVM 微基准" not in section, "§11.1 leaked into §11.3 input")
    return section


def parse_subsection_table(block: str, subsection: str) -> pd.DataFrame:
    table_lines = [line for line in block.splitlines() if line.startswith("|")]
    require(len(table_lines) >= 3, f"no Markdown table found in §{subsection}")
    header = split_markdown_row(table_lines[0])
    rows = [
        split_markdown_row(line)
        for line in table_lines[2:]
        if not re.fullmatch(r"[|:\- ]+", line)
    ]
    require(all(len(row) == len(header) for row in rows), f"malformed table in §{subsection}")
    return pd.DataFrame(rows, columns=header)


def build_tidy_data(report_path: Path) -> pd.DataFrame:
    section = extract_section(report_path)
    heading_matches = list(
        re.finditer(r"^#### (11\.3\.[12]) .+$", section, flags=re.MULTILINE)
    )
    require(len(heading_matches) == 2, "expected exactly §11.3.1 and §11.3.2")

    records: list[dict[str, object]] = []
    for index, match in enumerate(heading_matches):
        subsection = match.group(1)
        model = MODEL_BY_SUBSECTION[subsection]
        block_end = (
            heading_matches[index + 1].start()
            if index + 1 < len(heading_matches)
            else len(section)
        )
        table = parse_subsection_table(section[match.end() : block_end], subsection)
        required_columns = {
            "请求",
            "C",
            "Device TTFT",
            "UVM TTFT",
            "Device TPOT",
            "UVM TPOT",
            "Device E2E",
            "UVM E2E",
        }
        require(required_columns.issubset(table.columns), f"columns differ in §{subsection}")

        available: dict[tuple[str, int], pd.Series] = {}
        for _, source_row in table.iterrows():
            workload = str(source_row["请求"])
            concurrency = int(source_row["C"])
            device_ttft = parse_number(str(source_row["Device TTFT"]))
            if device_ttft is not None:
                available[(workload, concurrency)] = source_row

        expected = EXPECTED_CONFIGS[model]
        require(
            set(available) == set(expected),
            f"Device-resident non-OOM configurations differ for {model}: {sorted(available)}",
        )
        for workload, concurrency in expected:
            source_row = available[(workload, concurrency)]
            for metric in METRICS:
                device_ms = parse_number(str(source_row[f"Device {metric}"]))
                uvm_ms = parse_number(str(source_row[f"UVM {metric}"]))
                require(
                    device_ms is not None and uvm_ms is not None,
                    f"unexpected OOM in plotted cell {model}/{workload}/C{concurrency}/{metric}",
                )
                require(
                    device_ms > 0 and uvm_ms > 0,
                    f"non-positive latency in {model}/{workload}/C{concurrency}/{metric}",
                )
                records.append(
                    {
                        "model": model,
                        "workload": workload,
                        "concurrency": concurrency,
                        "metric": metric,
                        "device_ms": device_ms,
                        "uvm_ms": uvm_ms,
                        "slowdown": uvm_ms / device_ms,
                        "source_section": f"REPORT.md §{subsection}",
                        "validity": "valid_hash_match_single_run",
                    }
                )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 30, f"expected 30 valid observations, found {len(data)}")
    require(data["slowdown"].map(math.isfinite).all(), "non-finite slowdown found")
    require((data["slowdown"] > 0).all(), "non-positive slowdown found")
    return data


def write_csv(data: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False, float_format="%.6f")


def validate_serialized_csv(data: pd.DataFrame) -> None:
    expected_columns = [
        "model",
        "workload",
        "concurrency",
        "metric",
        "device_ms",
        "uvm_ms",
        "slowdown",
        "source_section",
        "validity",
    ]
    require(list(data.columns) == expected_columns, "serialized CSV columns differ")
    require(len(data) == 30, "serialized CSV row count differs")
    require(
        set(data["source_section"])
        == {"REPORT.md §11.3.1", "REPORT.md §11.3.2"},
        "CSV contains data outside §11.3.1/§11.3.2",
    )
    require(
        set(data["validity"]) == {"valid_hash_match_single_run"},
        "invalid observations entered the CSV",
    )
    recomputed = data["uvm_ms"] / data["device_ms"]
    require(
        np.allclose(recomputed, data["slowdown"], rtol=0, atol=5e-7),
        "serialized slowdowns differ from source-millisecond ratios",
    )


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def multiplier_tick(value: float, _position: int) -> str:
    return f"{value:g}×"


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    panels = [
        ("DeepSeek-V2-Lite BF16", "(a) DeepSeek-V2-Lite BF16"),
        ("Qwen3-30B-A3B FP8", "(b) Qwen3-30B-A3B FP8"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)

    for ax, (model, panel_title) in zip(axes, panels, strict=True):
        order = EXPECTED_CONFIGS[model]
        labels = [f"{workload}/C{concurrency}" for workload, concurrency in order]
        base_x = np.arange(len(order), dtype=float)
        model_data = data[data["model"] == model]

        for metric in METRICS:
            style = METRIC_STYLE[metric]
            metric_data = model_data[model_data["metric"] == metric].set_index(
                ["workload", "concurrency"]
            )
            values = np.array(
                [float(metric_data.loc[config, "slowdown"]) for config in order]
            )
            ax.scatter(
                base_x + float(style["offset"]),
                values,
                s=29,
                color=str(style["color"]),
                marker=str(style["marker"]),
                edgecolors=str(style["color"]),
                linewidths=0.8,
                zorder=3,
            )

        short_c16 = model_data[
            (model_data["workload"] == "Short")
            & (model_data["concurrency"] == 16)
            & (model_data["metric"] == "TTFT")
        ]
        require(len(short_c16) == 1, f"missing Short/C16 TTFT annotation for {model}")
        slowdown = float(short_c16.iloc[0]["slowdown"])
        annotation_label = TTFT_ANNOTATION_BY_MODEL[model]
        require(
            abs(slowdown - annotation_label) < 0.01,
            f"computed TTFT ratio is inconsistent with the reported label for {model}",
        )
        annotation_x = base_x[1] + float(METRIC_STYLE["TTFT"]["offset"])
        ax.annotate(
            f"{annotation_label:.2f}×",
            xy=(annotation_x, slowdown),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=str(METRIC_STYLE["TTFT"]["color"]),
            zorder=4,
        )

        ax.axhline(
            1.0,
            color="#777777",
            linestyle=(0, (3.0, 2.2)),
            linewidth=1.0,
            zorder=1,
        )
        ax.set_yscale("log")
        ax.set_ylim(0.82, 40.0)
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 40]))
        ax.yaxis.set_major_formatter(FuncFormatter(multiplier_tick))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(base_x, labels=labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Workload / Concurrency")
        ax.set_title(panel_title, pad=5.0)
        ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        ax.grid(axis="y", which="minor", visible=False)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Slowdown over device-resident (×)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=str(METRIC_STYLE[metric]["marker"]),
            markerfacecolor=str(METRIC_STYLE[metric]["color"]),
            markeredgecolor=str(METRIC_STYLE[metric]["color"]),
            markeredgewidth=0.8,
            markersize=5.4,
            label=metric,
        )
        for metric in METRICS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.105, right=0.992, bottom=0.285, top=0.78, wspace=0.16)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Stock UVM demand-paging slowdown",
        "Subject": "Real-model latency slowdown relative to device-resident allocation",
        "Creator": SCRIPT_PATH.name,
    }
    fig.savefig(pdf_path, format="pdf", metadata=metadata)
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        metadata={"Software": SCRIPT_PATH.name},
    )
    plt.close(fig)


def write_caption(data: pd.DataFrame, caption_path: Path) -> None:
    ranges = data.groupby("metric", observed=True)["slowdown"].agg(["min", "max"])
    caption = (
        "Performance overhead of Stock UVM demand paging relative to device-resident "
        "allocation for DeepSeek-V2-Lite BF16 and Qwen3-30B-A3B FP8. Across the "
        f"evaluated workloads, Stock UVM increases TTFT by "
        f"{ranges.loc['TTFT', 'min']:.2f}–"
        f"{max(TTFT_ANNOTATION_BY_MODEL.values()):.2f}×, TPOT by "
        f"{ranges.loc['TPOT', 'min']:.2f}–{ranges.loc['TPOT', 'max']:.2f}×, and "
        f"end-to-end latency by {ranges.loc['E2E', 'min']:.2f}–"
        f"{ranges.loc['E2E', 'max']:.2f}×. Each point is a ratio of request-level P50 "
        "latencies in milliseconds from one independent cold process per cell under "
        "eager direct-batch execution; no error bars are shown. C1 and C16 denote "
        "concurrency 1 and 16. All plotted cells have matching output-token hashes; "
        "configurations where device-resident execution OOMs are excluded. Results are "
        "limited to one RTX 5090, frozen requests, and unhinted Stock UVM without "
        "prefetch or memory advice, and do not represent tuned UVM or AdaptMoe. Stock "
        "UVM demand paging substantially increases TTFT, TPOT, and end-to-end latency, "
        "especially under higher concurrency."
    )
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    csv_path = args.csv.resolve()
    pdf_path = args.pdf.resolve()
    png_path = args.png.resolve()
    caption_path = args.caption.resolve()
    require(report_path.is_file(), f"report not found: {report_path}")

    source_data = build_tidy_data(report_path)
    write_csv(source_data, csv_path)
    # Plot from the serialized tidy CSV, not from manually entered values.
    plot_data = pd.read_csv(csv_path)
    validate_serialized_csv(plot_data)
    render_figure(plot_data, pdf_path, png_path)
    write_caption(plot_data, caption_path)

    print(f"[OK] source: {report_path} §11.3 only")
    print(f"[OK] CSV: {csv_path} ({len(plot_data)} valid rows)")
    print(f"[OK] PDF: {pdf_path}")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] caption: {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
