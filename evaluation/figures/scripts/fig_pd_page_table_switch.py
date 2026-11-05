#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot Experiment B page-table view switching measurements."""

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
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MultipleLocator, NullFormatter


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_pd_page_table_switch"

DEFAULT_REPORT = REPO_ROOT / "evaluation/data/processed/REPORT.md"
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

DIRECTIONS = ("Prefill→Decode", "Decode→Prefill")
DIRECTION_COLORS = {
    "Prefill→Decode": "#0072B2",
    "Decode→Prefill": "#E69F00",
}
METHOD_STYLES = {
    "Parent/PDE": {"color": "#0072B2", "hatch": "//"},
    "Leaf-PTE": {"color": "#D55E00", "hatch": "\\\\"},
}
COMPONENT_COLORS = {
    "PTE map": "#0072B2",
    "CPU unmap": "#E69F00",
    "Preflight": "#009E73",
    "Build/sort": "#CC79A7",
    "Metadata": "#56B4E9",
    "Residual": "#777777",
}
CSV_COLUMNS = (
    "panel",
    "direction",
    "category",
    "statistic",
    "value_ms",
    "total_ms",
    "percentage",
    "sample_n",
    "sample_nature",
    "measurement_scope",
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


def parse_ms(cell: str, label: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?) ms", cell.strip())
    require(match is not None, f"invalid {label}: {cell}")
    value = float(match.group(1))
    require(value > 0 and math.isfinite(value), f"non-positive {label}: {cell}")
    return value


def direction_from_operation(operation: str) -> str:
    for direction in DIRECTIONS:
        if operation.startswith(direction):
            return direction
    raise ValueError(f"unexpected operation: {operation}")


def parse_table(section: str, expected_header: list[str], expected_rows: int) -> pd.DataFrame:
    lines = section.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if split_markdown_row(line) == expected_header),
        -1,
    )
    require(start >= 0, f"table header not found: {expected_header}")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    require(len(table_lines) == expected_rows + 2, "table row count differs")
    rows = [split_markdown_row(line) for line in table_lines[2:]]
    require(all(len(row) == len(expected_header) for row in rows), "malformed table row")
    return pd.DataFrame(rows, columns=expected_header)


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    experiment = extract_between(
        report,
        "## 5. 实验 B：P/D 快速页表切换",
        "## 6. 实验 C：物理显存生命周期",
    )
    section_51 = extract_between(
        experiment,
        "### 5.1 8 MiB parent/PDE 1,000 循环",
        "### 5.2 cold80 全规模 4 KiB 叶级 PTE 点",
    )
    section_52 = extract_between(
        experiment,
        "### 5.2 cold80 全规模 4 KiB 叶级 PTE 点",
        "### 5.3 同几何 parent 路径参考",
    )
    section_53 = extract_between(
        experiment,
        "### 5.3 同几何 parent 路径参考",
        "### 5.4 Full-model Qwen 失败点",
    )
    section_55 = experiment[experiment.find("### 5.5 B 的结论"):]

    require("测量的是页表视图转换，不是请求暂停时间" in section_51,
            "8 MiB scope statement differs")
    require("单次运行，无统计重复" in section_52, "Leaf sample nature differs")
    require("不包含 backing 分配" in section_52, "backing-allocation exclusion missing")
    require("也不是 18.119 GB 数据复制时间" in section_52, "copy exclusion missing")
    require("不能把上述倍率当作正式配对统计结果" in section_53,
            "descriptive-ratio limitation missing")
    require("25%/50%/75%/100%" in section_55, "missing full-model matrix limitation")

    records: list[dict[str, object]] = []
    table_51 = parse_table(
        section_51,
        ["操作", "样本", "Min", "Mean", "P50", "P95", "Max"],
        2,
    )
    for _, row in table_51.iterrows():
        direction = direction_from_operation(str(row["操作"]))
        sample_n = int(str(row["样本"]).replace(",", ""))
        require(sample_n == 1000, "8 MiB sample count differs")
        for statistic in ("Min", "P50", "P95", "Max"):
            records.append(
                {
                    "panel": "a",
                    "direction": direction,
                    "category": "8 MiB Parent/PDE",
                    "statistic": statistic.lower(),
                    "value_ms": parse_ms(str(row[statistic]), statistic),
                    "total_ms": math.nan,
                    "percentage": math.nan,
                    "sample_n": sample_n,
                    "sample_nature": "1,000-cycle distribution summary",
                    "measurement_scope": "mapping-view transition; not request pause",
                    "source_section": "REPORT.md §5.1 Experiment B",
                    "validity": "valid_distribution_summary_not_independent_repeats",
                }
            )

    table_52 = parse_table(
        section_52,
        ["操作", "Outer wall", "Driver wall", "PTE map", "TLB invalidates", "Batch"],
        2,
    )
    leaf_values: dict[str, float] = {}
    leaf_pte_map = math.nan
    for _, row in table_52.iterrows():
        direction = direction_from_operation(str(row["操作"]))
        leaf_values[direction] = parse_ms(str(row["Outer wall"]), "Leaf outer wall")
        if direction == "Decode→Prefill":
            leaf_pte_map = parse_ms(str(row["PTE map"]), "PTE map")

    table_53 = parse_table(
        section_53,
        ["操作", "Count", "Min", "Mean", "Median", "Max"],
        2,
    )
    parent_values: dict[str, float] = {}
    parent_counts: dict[str, int] = {}
    for _, row in table_53.iterrows():
        direction = direction_from_operation(str(row["操作"]))
        parent_values[direction] = parse_ms(str(row["Median"]), "Parent median")
        parent_counts[direction] = int(row["Count"])

    for direction in DIRECTIONS:
        for category, value, sample_n, sample_nature in (
            ("Parent/PDE", parent_values[direction], parent_counts[direction],
             "median from a separate qualification trace"),
            ("Leaf-PTE", leaf_values[direction], 1, "single-sample synthetic qualification"),
        ):
            records.append(
                {
                    "panel": "b",
                    "direction": direction,
                    "category": category,
                    "statistic": "median" if category == "Parent/PDE" else "outer_wall",
                    "value_ms": value,
                    "total_ms": math.nan,
                    "percentage": math.nan,
                    "sample_n": sample_n,
                    "sample_nature": sample_nature,
                    "measurement_scope": "cold80 same geometry; different qualification runs",
                    "source_section": (
                        "REPORT.md §5.3 Experiment B"
                        if category == "Parent/PDE"
                        else "REPORT.md §5.2 Experiment B"
                    ),
                    "validity": "valid_descriptive_reference_not_formally_paired",
                }
            )

    component_patterns = {
        "CPU unmap": r"([0-9]+(?:\.[0-9]+)?) ms CPU unmap",
        "Preflight": r"([0-9]+(?:\.[0-9]+)?) ms preflight",
        "Build/sort": r"([0-9]+(?:\.[0-9]+)?) ms build/sort",
        "Metadata": r"([0-9]+(?:\.[0-9]+)?) ms metadata",
    }
    total = leaf_values["Decode→Prefill"]
    components = {"PTE map": leaf_pte_map}
    for component, pattern in component_patterns.items():
        match = re.search(pattern, section_52)
        require(match is not None, f"component not found: {component}")
        components[component] = float(match.group(1))
    residual = total - sum(components.values())
    require(residual > 0 and math.isfinite(residual), "invalid residual")
    components["Residual"] = residual
    require(math.isclose(sum(components.values()), total, abs_tol=1e-9),
            "component decomposition does not sum to outer wall")
    for component, value in components.items():
        records.append(
            {
                "panel": "c",
                "direction": "Decode→Prefill",
                "category": component,
                "statistic": "outer_wall_component",
                "value_ms": value,
                "total_ms": total,
                "percentage": value / total * 100.0,
                "sample_n": 1,
                "sample_nature": "single-sample synthetic qualification decomposition",
                "measurement_scope": "excludes backing allocation and 18.119 GB data copy",
                "source_section": "REPORT.md §5.2 Experiment B",
                "validity": "valid_component_or_computed_residual",
            }
        )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 18, "expected 18 tidy records")
    require(math.isclose(residual, 0.353, abs_tol=1e-9), "residual differs")
    return data.loc[:, CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.6f", na_rep="")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == list(CSV_COLUMNS), "CSV columns differ")
    require(len(serialized) == 18, "serialized row count differs")
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.3,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.2,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def value(data: pd.DataFrame, panel: str, direction: str, category: str, statistic: str) -> float:
    selected = data[
        (data["panel"] == panel)
        & (data["direction"] == direction)
        & (data["category"] == category)
        & (data["statistic"] == statistic)
    ]
    require(len(selected) == 1, f"value lookup differs: {panel}/{direction}/{category}/{statistic}")
    return float(selected.iloc[0]["value_ms"])


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.35), gridspec_kw={"width_ratios": [0.9, 1.0, 1.35]})

    # (a) Distribution summaries without inventing quartiles.
    axis = axes[0]
    x_positions = np.arange(len(DIRECTIONS), dtype=float)
    for x, direction in zip(x_positions, DIRECTIONS, strict=True):
        minimum = value(data, "a", direction, "8 MiB Parent/PDE", "min")
        p50 = value(data, "a", direction, "8 MiB Parent/PDE", "p50")
        p95 = value(data, "a", direction, "8 MiB Parent/PDE", "p95")
        maximum = value(data, "a", direction, "8 MiB Parent/PDE", "max")
        color = DIRECTION_COLORS[direction]
        axis.vlines(x, minimum, maximum, color=color, linewidth=0.9, zorder=2)
        axis.hlines([minimum, maximum], x - 0.06, x + 0.06, color=color, linewidth=0.9, zorder=2)
        axis.vlines(x, p50, p95, color=color, linewidth=3.2, zorder=3)
        axis.scatter([x], [p50], s=27, color=color, edgecolor="white", linewidth=0.55, zorder=4)
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylim(0, 1.05)
    axis.set_xticks(x_positions, ["Prefill→\nDecode", "Decode→\nPrefill"])
    axis.set_ylabel("Switch latency (ms)")
    axis.set_title("(a) 8 MiB, 1,000 cycles", loc="left", pad=3)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    summary_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="#555555", markersize=4.2, label="P50"),
        Line2D([0], [0], color="#555555", linewidth=3.0, label="P50–P95"),
        Line2D([0], [0], color="#555555", linewidth=0.9, label="Min–Max"),
    ]
    axis.legend(handles=summary_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                ncol=1, frameon=False, handlelength=1.4, borderaxespad=0.1)

    # (b) Same geometry, but separate qualification runs.
    axis = axes[1]
    width = 0.30
    for method_index, method in enumerate(("Parent/PDE", "Leaf-PTE")):
        xs = x_positions + (-width / 2 if method_index == 0 else width / 2)
        ys = [value(data, "b", direction, method,
                    "median" if method == "Parent/PDE" else "outer_wall")
              for direction in DIRECTIONS]
        style = METHOD_STYLES[method]
        axis.bar(xs, ys, width=width, color=style["color"], edgecolor="#333333",
                 linewidth=0.6, hatch=style["hatch"], label=method, zorder=3)
    ratios = (
        value(data, "b", "Prefill→Decode", "Leaf-PTE", "outer_wall")
        / value(data, "b", "Prefill→Decode", "Parent/PDE", "median"),
        value(data, "b", "Decode→Prefill", "Leaf-PTE", "outer_wall")
        / value(data, "b", "Decode→Prefill", "Parent/PDE", "median"),
    )
    leaf_xs = x_positions + width / 2
    leaf_ys = [value(data, "b", direction, "Leaf-PTE", "outer_wall") for direction in DIRECTIONS]
    for x, y, ratio in zip(leaf_xs, leaf_ys, ratios, strict=True):
        axis.annotate(f"{ratio:.1f}×", (x, y), xytext=(0, 3), textcoords="offset points",
                      ha="center", va="bottom", fontsize=6.4, fontweight="bold",
                      color=METHOD_STYLES["Leaf-PTE"]["color"])
    axis.set_yscale("log")
    axis.set_ylim(15, 500)
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(x_positions, ["Prefill→\nDecode", "Decode→\nPrefill"])
    axis.set_ylabel("Latency (ms)")
    axis.set_title("(b) Parent vs Leaf-PTE", loc="left", pad=3)
    axis.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)

    # (c) Preserve every named component and the computed residual.
    axis = axes[2]
    panel_c = data[data["panel"] == "c"].set_index("category")
    component_order = tuple(COMPONENT_COLORS)
    left = 0.0
    for component in component_order:
        component_value = float(panel_c.loc[component, "value_ms"])
        axis.barh([0], [component_value], left=left, height=0.42,
                  color=COMPONENT_COLORS[component], edgecolor="white", linewidth=0.4,
                  label=component, zorder=3)
        if component in ("PTE map", "CPU unmap"):
            percentage = float(panel_c.loc[component, "percentage"])
            axis.text(left + component_value / 2, 0, f"{component}\n{percentage:.1f}%",
                      ha="center", va="center", fontsize=5.8,
                      color="white" if component == "PTE map" else "#222222",
                      fontweight="bold", linespacing=0.9, zorder=4)
        left += component_value
    total = float(panel_c["total_ms"].iloc[0])
    require(math.isclose(left, total, abs_tol=1e-6), "rendered stack differs from total")
    axis.set_xlim(0, 305)
    axis.set_ylim(-0.55, 0.55)
    axis.set_yticks([0], ["D→P"])
    axis.set_xlabel("Latency (ms)")
    axis.set_title("(c) Leaf-PTE breakdown", loc="left", pad=3)
    axis.xaxis.set_major_locator(MultipleLocator(100))
    component_handles = [Patch(facecolor=color, edgecolor="none", label=component)
                         for component, color in COMPONENT_COLORS.items()]
    axis.legend(handles=component_handles, loc="upper center", bbox_to_anchor=(0.5, -0.31),
                ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8,
                labelspacing=0.35, borderaxespad=0)

    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="y" if axis is axes[2] else "y", color="#DEDEDE", linewidth=0.55)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.34, top=0.90, wspace=0.38)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Prefill/decode page-table view switching",
        "Subject": "Parent/PDE switching, Leaf-PTE reference, and Leaf-PTE decomposition",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(png_path, format="png", dpi=300, metadata={"Software": SCRIPT_PATH.name})
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "Prefill/decode page-table view switching. (a) For the 8 MiB, 1,000-cycle "
        "qualification, points show P50, thick whiskers span P50–P95, and thin whiskers "
        "span Min–Max; these are distribution summaries, not quartiles. The loop measures "
        "mapping-view conversion rather than request pause time. (b) The cold80 Parent/PDE "
        "medians and Leaf-PTE outer-wall values come from different qualification runs; "
        "the 2.8× and 11.0× ratios are descriptive references, not formally paired "
        "statistics, and each Leaf-PTE value is a single sample. A full-model paired "
        "25/50/75/100% matrix has not been completed. (c) Decode-to-Prefill Leaf-PTE time "
        "is decomposed without discarding the computed residual; PTE mapping accounts for "
        "approximately 76% of outer-wall time. Measurements exclude backing allocation and "
        "do not represent copying 18.119 GB of data. Preconstructed parent-level views avoid "
        "the scale-dependent cost of rebuilding tens of thousands of leaf PTE mappings."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    require(report_path.is_file(), f"report not found: {report_path}")
    data = build_data(report_path)
    plot_data = write_and_reload_csv(data, args.csv.resolve())
    render_figure(plot_data, args.pdf.resolve(), args.png.resolve())
    write_caption(args.caption.resolve())
    residual = float(plot_data[(plot_data["panel"] == "c") &
                               (plot_data["category"] == "Residual")]["value_ms"].iloc[0])
    print(f"[OK] sole source: {report_path}")
    print(f"[OK] CSV: {args.csv.resolve()} ({len(plot_data)} rows)")
    print(f"[OK] PDF: {args.pdf.resolve()}")
    print(f"[OK] PNG: {args.png.resolve()}")
    print(f"[OK] caption: {args.caption.resolve()}")
    print(f"[SUMMARY] Decode→Prefill residual: {residual:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
