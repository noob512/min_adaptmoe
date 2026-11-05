#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot E2 end-to-end latency for the six available system configurations."""

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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_e2e_six_systems"

DEFAULT_REPORT = (
    REPO_ROOT
    / "evaluation/data/processed/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md"
)
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

MODEL = "DeepSeek-V2-Lite-Chat BF16"
WORKLOADS = ("Short", "Ordinary", "RAG", "Long")
CONCURRENCIES = (1, 16)
MECHANISMS = (
    "AdaptMoe",
    "all-GPU eager",
    "pinned-UVA eager",
    "copy-on-forward eager",
    "KTransformers",
    "MoE-Infinity",
)
SOURCE_COLUMNS = {
    "ours，全覆盖双表+Graph": "AdaptMoe",
    "all-GPU eager": "all-GPU eager",
    "pinned-UVA eager": "pinned-UVA eager",
    "copy-on-forward eager": "copy-on-forward eager",
    "KTransformers": "KTransformers",
    "MoE-Infinity": "MoE-Infinity",
}
STYLES = {
    "AdaptMoe": {"color": "#0072B2", "marker": "o", "size": 6.5},
    "all-GPU eager": {"color": "#4D4D4D", "marker": "s", "size": 5.2},
    "pinned-UVA eager": {"color": "#E69F00", "marker": "^", "size": 5.5},
    "copy-on-forward eager": {"color": "#009E73", "marker": "D", "size": 5.0},
    "KTransformers": {"color": "#CC79A7", "marker": "v", "size": 5.5},
    "MoE-Infinity": {"color": "#D55E00", "marker": "x", "size": 6.0},
}
OFFSETS = dict(zip(MECHANISMS, np.linspace(-0.27, 0.27, len(MECHANISMS)), strict=True))

CSV_COLUMNS = [
    "model",
    "workload",
    "concurrency",
    "mechanism",
    "e2e_ms",
    "status",
    "exact_token_match",
    "formal_ranking_eligible",
    "note_symbol",
    "sample_n",
    "source_section",
    "validity",
]


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


def section(report: str, start_heading: str, end_heading: str) -> str:
    start = report.find(start_heading)
    end = report.find(end_heading, start + len(start_heading))
    require(start >= 0 and end > start, f"section boundaries not found: {start_heading}")
    return report[start:end]


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_markdown_table(block: str) -> pd.DataFrame:
    lines = block.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith("| 场景 | ours，全覆盖双表+Graph |")),
        -1,
    )
    require(start >= 0, "E2 §5.2 primary table not found")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    require(len(table_lines) == 10, "E2 §5.2 table must contain one header and eight data rows")
    header = split_markdown_row(table_lines[0])
    require(header == ["场景", *SOURCE_COLUMNS.keys()], "E2 §5.2 table header differs")
    rows = [split_markdown_row(line) for line in table_lines[2:]]
    require(all(len(row) == len(header) for row in rows), "malformed E2 §5.2 table")
    return pd.DataFrame(rows, columns=header)


def parse_cell(cell: str) -> tuple[float, str, str]:
    value = cell.strip()
    if value == "OOM":
        return math.nan, "OOM", ""
    if value in {"—", "N/A", ""}:
        return math.nan, "missing", ""
    symbols = "".join(symbol for symbol in "†§" if symbol in value)
    numeric = re.sub(r"[†§,\s]", "", value)
    require(re.fullmatch(r"\d+(?:\.\d+)?", numeric) is not None, f"invalid cell: {cell}")
    parsed = float(numeric)
    require(parsed > 0 and math.isfinite(parsed), f"invalid latency: {cell}")
    return parsed, "ok", symbols


def validity_fields(
    mechanism: str, workload: str, concurrency: int, status: str, symbol: str
) -> tuple[bool | None, bool, str]:
    if status == "OOM":
        return None, False, "oom_not_numeric"
    if status == "missing":
        return None, False, "missing_not_numeric"
    if mechanism == "AdaptMoe" and workload == "RAG":
        require(symbol == "†", "AdaptMoe RAG point lacks dagger")
        return False, False, "completed_hash_mismatch_excluded_from_ranking"
    if mechanism == "KTransformers":
        if workload == "RAG" and concurrency == 16:
            require(symbol == "§", "KTransformers RAG/C16 lacks section mark")
            return None, False, "raw_diagnostic_missing_case_verification_receipt"
        return None, False, "completed_without_cross_framework_generation_gate"
    if mechanism == "MoE-Infinity":
        return None, False, "completed_without_formal_cross_framework_generation_gate"
    if workload == "RAG":
        return True, False, "valid_native_path_but_not_adaptmoe_exact_token_pair"
    return True, True, "valid_exact_token_comparison_point"


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    section_51 = section(report, "### 5.1 ", "### 5.2 ")
    section_52 = section(report, "### 5.2 ", "### 5.5 ")
    section_55 = section(report, "### 5.5 ", "#### 5.5.1 ")
    section_18 = section(report, "## 18. ", "## 19. ")

    require("每个场景一次独立冷进程" in section_51, "§5.1 cold-process statement missing")
    require("RAG/C1" in section_51 and "RAG/C16" in section_51, "§5.1 RAG audit missing")
    require("跨臂hash不同" in section_51, "§5.1 hash mismatch statement missing")
    require("不是单项机制消融" in section_52, "§5.2 ablation limitation missing")
    require("非equal-HBM" in section_52, "§5.2 equal-HBM limitation missing")
    require("TTFT和E2E不能解释为双Graph" in section_55, "§5.5 E2E limitation missing")
    require("不能声称ours在所有模型、请求和系统上全局最优" in section_18, "§18 ranking limit missing")
    require("不能把E2的RAG/C1和RAG/C16写入跨系统exact-token性能排名" in section_18, "§18 RAG limit missing")

    source_table = parse_markdown_table(section_52)
    records: list[dict[str, object]] = []
    for _, source_row in source_table.iterrows():
        match = re.fullmatch(r"(Short|Ordinary|RAG|Long)/C(1|16)", source_row["场景"])
        require(match is not None, f"invalid scenario: {source_row['场景']}")
        workload = match.group(1)
        concurrency = int(match.group(2))
        for source_column, mechanism in SOURCE_COLUMNS.items():
            e2e_ms, status, symbol = parse_cell(source_row[source_column])
            exact_match, eligible, validity = validity_fields(
                mechanism, workload, concurrency, status, symbol
            )
            records.append(
                {
                    "model": MODEL,
                    "workload": workload,
                    "concurrency": concurrency,
                    "mechanism": mechanism,
                    "e2e_ms": e2e_ms,
                    "status": status,
                    "exact_token_match": exact_match,
                    "formal_ranking_eligible": eligible,
                    "note_symbol": symbol,
                    "sample_n": 1 if status == "ok" else 0,
                    "source_section": (
                        "RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md E2 §5.2; "
                        "validity from E2 §5.1, §5.5, and §18"
                    ),
                    "validity": validity,
                }
            )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 48, "expected 48 tidy rows")
    require(set(data["workload"]) == set(WORKLOADS), "workload coverage differs")
    require(set(data["concurrency"]) == set(CONCURRENCIES), "concurrency coverage differs")
    require(set(data["mechanism"]) == set(MECHANISMS), "mechanism coverage differs")
    require((data["status"] == "ok").sum() == 37, "expected 37 numeric cells")
    require((data["status"] == "OOM").sum() == 3, "expected three OOM cells")
    require((data["status"] == "missing").sum() == 8, "expected eight missing cells")
    require(
        data.loc[data["status"] != "ok", "e2e_ms"].isna().all(),
        "OOM or missing value was numericized",
    )
    return data[CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.2f", na_rep="")
    serialized = pd.read_csv(path, keep_default_na=False)
    require(list(serialized.columns) == CSV_COLUMNS, "serialized CSV columns differ")
    require(len(serialized) == 48, "serialized CSV row count differs")
    numeric = pd.to_numeric(serialized["e2e_ms"], errors="coerce")
    require(numeric[serialized["status"] != "ok"].isna().all(), "non-valid CSV row is numeric")
    serialized["e2e_ms"] = numeric
    for column in ("exact_token_match", "formal_ranking_eligible"):
        serialized[column] = serialized[column].map(
            {"True": True, "False": False, "": None}
        )
    return serialized


def lookup(data: pd.DataFrame, workload: str, concurrency: int, mechanism: str) -> pd.Series:
    selected = data[
        (data["workload"] == workload)
        & (data["concurrency"] == concurrency)
        & (data["mechanism"] == mechanism)
    ]
    require(len(selected) == 1, f"lookup differs: {workload}/C{concurrency}/{mechanism}")
    return selected.iloc[0]


def exact_speedup(data: pd.DataFrame, workload: str, concurrency: int, baseline: str) -> float:
    adapt = lookup(data, workload, concurrency, "AdaptMoe")
    other = lookup(data, workload, concurrency, baseline)
    require(adapt["status"] == "ok" and other["status"] == "ok", "speedup point is unavailable")
    require(adapt["exact_token_match"] is True, "AdaptMoe speedup point is not exact-token")
    require(other["exact_token_match"] is True, "baseline speedup point is not exact-token")
    return float(other["e2e_ms"] / adapt["e2e_ms"])


def verify_published_ranges(data: pd.DataFrame) -> None:
    exact_workloads = ("Short", "Ordinary", "Long")
    pinned = [
        exact_speedup(data, workload, concurrency, "pinned-UVA eager")
        for concurrency in CONCURRENCIES
        for workload in exact_workloads
    ]
    copy = [
        exact_speedup(data, workload, concurrency, "copy-on-forward eager")
        for concurrency in CONCURRENCIES
        for workload in exact_workloads
    ]
    require((round(min(pinned), 2), round(max(pinned), 2)) == (2.09, 11.56), "pinned range differs")
    require((round(min(copy), 2), round(max(copy), 2)) == (2.01, 8.96), "copy range differs")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.3,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def latency_tick(value: float, _position: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    x_base = np.arange(len(WORKLOADS), dtype=float)
    valid_values = data.loc[data["status"] == "ok", "e2e_ms"].to_numpy(dtype=float)
    y_min = float(valid_values.min()) / 1.75
    y_max = float(valid_values.max()) * 1.75
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=True)

    for axis, concurrency, panel in zip(axes, CONCURRENCIES, ("(a)", "(b)"), strict=True):
        panel_data = data[data["concurrency"] == concurrency]
        for workload_index, workload in enumerate(WORKLOADS):
            for mechanism in MECHANISMS:
                row = lookup(panel_data, workload, concurrency, mechanism)
                x = workload_index + OFFSETS[mechanism]
                style = STYLES[mechanism]
                if row["status"] == "OOM":
                    axis.text(
                        x,
                        0.955,
                        "OOM",
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=6.5,
                        fontweight="bold",
                        color=style["color"],
                        clip_on=False,
                    )
                    continue
                if row["status"] == "missing":
                    continue

                hollow = mechanism == "AdaptMoe" and workload == "RAG"
                facecolor = "white" if hollow else style["color"]
                if style["marker"] == "x":
                    facecolor = "none"
                scatter_style = {
                    "s": style["size"] ** 2,
                    "marker": style["marker"],
                    "linewidths": 1.15,
                    "zorder": 3,
                }
                if style["marker"] == "x":
                    scatter_style["color"] = style["color"]
                else:
                    scatter_style["facecolors"] = facecolor
                    scatter_style["edgecolors"] = style["color"]
                axis.scatter([x], [row["e2e_ms"]], **scatter_style)
                if row["note_symbol"]:
                    axis.annotate(
                        row["note_symbol"],
                        (x, row["e2e_ms"]),
                        xytext=(4.0, 2.5),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=7.0,
                        fontweight="bold",
                        color=style["color"],
                    )

        axis.set_yscale("log")
        axis.set_ylim(y_min, y_max)
        axis.set_xlim(-0.55, len(WORKLOADS) - 0.45)
        axis.set_xticks(x_base, WORKLOADS)
        axis.set_xlabel("Workload")
        axis.set_ylabel("End-to-end latency (ms)")
        axis.set_title(f"{panel} Concurrency {concurrency}", loc="left", pad=3.0)
        axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        axis.yaxis.set_major_formatter(FuncFormatter(latency_tick))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.55, alpha=0.85)
        axis.grid(axis="y", which="minor", visible=False)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    c1_long_pinned = exact_speedup(data, "Long", 1, "pinned-UVA eager")
    c1_long_copy = exact_speedup(data, "Long", 1, "copy-on-forward eager")
    c16_short_pinned = exact_speedup(data, "Short", 16, "pinned-UVA eager")
    c16_short_copy = exact_speedup(data, "Short", 16, "copy-on-forward eager")
    axes[0].text(
        0.97,
        0.055,
        f"Long exact-token: {c1_long_pinned:.2f}× vs pinned-UVA\n"
        f"{c1_long_copy:.2f}× vs copy",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#333333",
    )
    axes[1].text(
        0.50,
        0.055,
        f"Short exact-token: {c16_short_pinned:.2f}× vs pinned-UVA\n"
        f"{c16_short_copy:.2f}× vs copy",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#333333",
    )

    legend_handles = []
    for mechanism in MECHANISMS:
        style = STYLES[mechanism]
        marker_face = "none" if style["marker"] == "x" else style["color"]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=style["marker"],
                markersize=style["size"],
                markerfacecolor=marker_face,
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                label=mechanism,
            )
        )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.25,
    )
    figure.subplots_adjust(left=0.105, right=0.992, bottom=0.18, top=0.79, wspace=0.31)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "End-to-end latency across available system configurations",
        "Subject": "E2 single-sample system-level comparison",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(png_path, format="png", dpi=300, metadata={"Software": SCRIPT_PATH.name})
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "End-to-end latency across available system configurations on "
        "DeepSeek-V2-Lite-Chat BF16 at concurrency 1 and 16. Each point is a "
        "single sample from an independently launched cold process; no independent "
        "repeats are available, so no error bars are shown. The systems are not a "
        "complete equal-HBM comparison or a single-variable ablation: placement, "
        "offload granularity, and execution paths differ. Hollow AdaptMoe RAG markers "
        "with † completed their single-arm checks but have a token-hash mismatch with "
        "the native vLLM paths and are excluded from formal performance ranking. The "
        "KTransformers RAG/C16 point marked § is raw diagnostic data without a "
        "case-level verification receipt; no formal speedup is reported for "
        "KTransformers or MoE-Infinity. OOM and missing entries are not numericized. "
        "Representative AdaptMoe speedups against pinned-UVA and copy-on-forward are "
        "shown only for exact-token Long/C1 and Short/C16 pairs."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_statistics(data: pd.DataFrame) -> None:
    print(f"Numeric cells: {(data['status'] == 'ok').sum()}/48")
    print(f"OOM cells: {(data['status'] == 'OOM').sum()}")
    print(f"Missing cells: {(data['status'] == 'missing').sum()}")
    print("Exact-token pinned-UVA/AdaptMoe speedup: 2.09×–11.56× (6 pairs)")
    print("Exact-token copy/AdaptMoe speedup: 2.01×–8.96× (6 pairs)")


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    csv_path = args.csv.resolve()
    pdf_path = args.pdf.resolve()
    png_path = args.png.resolve()
    caption_path = args.caption.resolve()
    require(report_path.is_file(), f"report not found: {report_path}")

    data = build_data(report_path)
    verify_published_ranges(data)
    plot_data = write_and_reload_csv(data, csv_path)
    render_figure(plot_data, pdf_path, png_path)
    write_caption(caption_path)

    print(f"[OK] source: {report_path}")
    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] PDF: {pdf_path}")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] caption: {caption_path}")
    print_statistics(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
