#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot E2E latency normalized to AdaptMoE from the latest E0--E14 matrix."""

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
FIGURE_NAME = "fig_e2e_six_systems_normalized"

DEFAULT_REPORT = REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E14_CONCISE_Full.md"
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

MODEL_WORKLOADS = {
    "DeepSeek-V2-Lite BF16": ("Short", "Ordinary", "RAG", "Long"),
    "Qwen3-30B-A3B FP8": ("Short", "Ordinary", "Medium", "RAG", "Long"),
}
CONCURRENCIES = (1, 16)
MECHANISMS = (
    "AdaptMoE",
    "Native UVA",
    "UVA Dual-Graph",
    "Native Copy",
    "Copy Dual-Graph",
    "KTransformers",
)
SOURCE_MECHANISMS = {
    "DeepSeek-V2-Lite BF16": {
        "AdaptMoE 双阶段 Graph": "AdaptMoE",
        "Native UVA（eager Prefill + Decode Graph）": "Native UVA",
        "Native Copy eager": "Native Copy",
        "KTransformers 7T + Decode Graph": "KTransformers",
        "Native UVA 双阶段 Graph": "UVA Dual-Graph",
        "Native Copy 双阶段 Graph": "Copy Dual-Graph",
    },
    "Qwen3-30B-A3B FP8": {
        "AdaptMoE 双阶段 Graph": "AdaptMoE",
        "Native UVA eager": "Native UVA",
        "Native Copy eager": "Native Copy",
        "KTransformers 7T + Decode Graph": "KTransformers",
        "Native UVA 双阶段 Graph": "UVA Dual-Graph",
        "Native Copy 双阶段 Graph": "Copy Dual-Graph",
    },
}
EXPECTED_OOM = {
    ("DeepSeek-V2-Lite BF16", "RAG", 1, "Copy Dual-Graph"),
    ("Qwen3-30B-A3B FP8", "RAG", 16, "Copy Dual-Graph"),
}

# Color encodes the data-path family; fill distinguishes baseline from Dual-Graph.
STYLES = {
    "AdaptMoE": {"color": "#0072B2", "marker": "o", "filled": True, "size": 6.5},
    "Native UVA": {"color": "#E69F00", "marker": "^", "filled": False, "size": 5.5},
    "UVA Dual-Graph": {"color": "#E69F00", "marker": "^", "filled": True, "size": 5.5},
    "Native Copy": {"color": "#009E73", "marker": "s", "filled": False, "size": 5.2},
    "Copy Dual-Graph": {"color": "#009E73", "marker": "s", "filled": True, "size": 5.2},
    "KTransformers": {"color": "#7E57C2", "marker": "D", "filled": True, "size": 5.0},
}
Y_OFFSETS = {
    "AdaptMoE": 0.00,
    "Native UVA": -0.24,
    "UVA Dual-Graph": -0.12,
    "Native Copy": 0.12,
    "Copy Dual-Graph": 0.24,
    "KTransformers": 0.36,
}

CSV_COLUMNS = [
    "model",
    "workload",
    "concurrency",
    "mechanism",
    "e2e_ms",
    "adaptmoe_e2e_ms",
    "normalized_e2e",
    "offload_gib",
    "status",
    "geomean_normalized_e2e",
    "adaptmoe_lowest_workload_count",
    "panel_max_valid_ratio",
    "is_ratio_below_one",
    "is_panel_slowest_valid_baseline",
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


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def extract_model_block(report: str, model: str, end_heading: str) -> str:
    start_heading = f"## {model}"
    start = report.find(start_heading)
    end = report.find(end_heading, start + len(start_heading))
    require(start >= 0 and end > start, f"model section not found: {model}")
    return report[start:end]


def parse_model_table(block: str, model: str) -> pd.DataFrame:
    expected_header = [
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
    lines = block.splitlines()
    start = next((index for index, line in enumerate(lines) if split_markdown_row(line) == expected_header), -1)
    require(start >= 0, f"primary table header missing for {model}")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    expected_data_rows = len(MODEL_WORKLOADS[model]) * len(CONCURRENCIES) * len(MECHANISMS)
    require(len(table_lines) == expected_data_rows + 2, f"table row count differs for {model}")
    header = split_markdown_row(table_lines[0])
    rows = [split_markdown_row(line) for line in table_lines[2:]]
    require(all(len(row) == len(header) for row in rows), f"malformed row in {model}")
    return pd.DataFrame(rows, columns=header)


def parse_positive(cell: str, label: str) -> float:
    value = cell.strip().replace(",", "")
    require(re.fullmatch(r"\d+(?:\.\d+)?", value) is not None, f"invalid {label}: {cell}")
    parsed = float(value)
    require(parsed > 0 and math.isfinite(parsed), f"non-positive {label}: {cell}")
    return parsed


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    require("# 两模型 C1/C16 六机制完整结果" in report, "unexpected source document")
    require("总矩阵为 108 cells；有效完成 106，N/A/OOM 2" in report, "coverage statement differs")

    blocks = {
        "DeepSeek-V2-Lite BF16": extract_model_block(
            report, "DeepSeek-V2-Lite BF16", "## Qwen3-30B-A3B FP8"
        ),
        "Qwen3-30B-A3B FP8": extract_model_block(
            report, "Qwen3-30B-A3B FP8", "## 覆盖与失败"
        ),
    }
    records: list[dict[str, object]] = []
    for model, workloads in MODEL_WORKLOADS.items():
        table = parse_model_table(blocks[model], model)
        source_map = SOURCE_MECHANISMS[model]
        require(set(table["机制"]) == set(source_map), f"mechanism set differs for {model}")
        for _, row in table.iterrows():
            workload = str(row["场景"])
            concurrency = int(row["C"])
            source_mechanism = str(row["机制"])
            require(workload in workloads, f"unexpected workload for {model}: {workload}")
            require(concurrency in CONCURRENCIES, f"unexpected concurrency: {concurrency}")
            mechanism = source_map[source_mechanism]
            offload_gib = parse_positive(str(row["实际卸载 GiB"]), "offload GiB")
            status_text = str(row["状态"])
            if status_text.startswith("ok"):
                status = "ok"
                e2e_ms = parse_positive(str(row["E2E (ms)"]), "E2E")
                validity = "valid_single_sample"
            elif status_text.startswith("OOM"):
                status = "OOM"
                require(str(row["E2E (ms)"]) == "—", "OOM row contains E2E")
                e2e_ms = math.nan
                validity = "oom_capacity_not_numeric"
            else:
                raise ValueError(f"unexpected status for {model}/{workload}/C{concurrency}: {status_text}")
            records.append(
                {
                    "model": model,
                    "workload": workload,
                    "concurrency": concurrency,
                    "mechanism": mechanism,
                    "e2e_ms": e2e_ms,
                    "offload_gib": offload_gib,
                    "status": status,
                    "source_section": f"RESULTS_E0_E14_CONCISE_Full.md §{model} primary table",
                    "validity": validity,
                }
            )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 108, "expected 108 data cells")
    observed_oom = {
        (row.model, row.workload, int(row.concurrency), row.mechanism)
        for row in data[data["status"] == "OOM"].itertuples()
    }
    require(observed_oom == EXPECTED_OOM, "OOM coverage differs")
    require(data.loc[data["status"] == "OOM", "e2e_ms"].isna().all(), "OOM became numeric")

    normalized_groups: list[pd.DataFrame] = []
    for (model, workload, concurrency), group in data.groupby(
        ["model", "workload", "concurrency"], sort=False
    ):
        adapt = group[group["mechanism"] == "AdaptMoE"]
        require(len(adapt) == 1 and adapt.iloc[0]["status"] == "ok", "AdaptMoE baseline missing")
        adapt_e2e = float(adapt.iloc[0]["e2e_ms"])
        group = group.copy()
        group["adaptmoe_e2e_ms"] = adapt_e2e
        group["normalized_e2e"] = group["e2e_ms"] / adapt_e2e
        group.loc[group["mechanism"] == "AdaptMoE", "normalized_e2e"] = 1.0
        normalized_groups.append(group)
    data = pd.concat(normalized_groups, ignore_index=True)
    require(data.loc[data["status"] == "OOM", "normalized_e2e"].isna().all(), "OOM ratio exists")

    data["geomean_normalized_e2e"] = math.nan
    data["adaptmoe_lowest_workload_count"] = 0
    data["panel_max_valid_ratio"] = math.nan
    data["is_ratio_below_one"] = False
    data["is_panel_slowest_valid_baseline"] = False
    for (model, concurrency), panel in data.groupby(["model", "concurrency"], sort=False):
        valid = panel[panel["status"] == "ok"]
        baselines = valid[valid["mechanism"] != "AdaptMoE"]
        panel_max = float(baselines["normalized_e2e"].max())
        slowest_index = baselines["normalized_e2e"].idxmax()
        lowest_count = 0
        for workload in MODEL_WORKLOADS[model]:
            workload_rows = valid[valid["workload"] == workload]
            adapt_value = float(workload_rows.loc[workload_rows["mechanism"] == "AdaptMoE", "e2e_ms"].iloc[0])
            if math.isclose(adapt_value, float(workload_rows["e2e_ms"].min()), rel_tol=0, abs_tol=1e-12):
                lowest_count += 1
        panel_indices = panel.index
        data.loc[panel_indices, "adaptmoe_lowest_workload_count"] = lowest_count
        data.loc[panel_indices, "panel_max_valid_ratio"] = panel_max
        data.loc[slowest_index, "is_panel_slowest_valid_baseline"] = True
        below = baselines["normalized_e2e"] < 1.0
        data.loc[baselines.index[below], "is_ratio_below_one"] = True
        for mechanism in MECHANISMS:
            mechanism_rows = valid[valid["mechanism"] == mechanism]
            require(len(mechanism_rows) > 0, f"no valid rows for {model}/C{concurrency}/{mechanism}")
            geomean = float(np.exp(np.log(mechanism_rows["normalized_e2e"]).mean()))
            mechanism_indices = panel[panel["mechanism"] == mechanism].index
            data.loc[mechanism_indices, "geomean_normalized_e2e"] = geomean

    model_rank = {model: index for index, model in enumerate(MODEL_WORKLOADS)}
    workload_rank = {
        (model, workload): index
        for model, workloads in MODEL_WORKLOADS.items()
        for index, workload in enumerate(workloads)
    }
    mechanism_rank = {mechanism: index for index, mechanism in enumerate(MECHANISMS)}
    data["_model_rank"] = data["model"].map(model_rank)
    data["_workload_rank"] = [workload_rank[(m, w)] for m, w in zip(data["model"], data["workload"])]
    data["_mechanism_rank"] = data["mechanism"].map(mechanism_rank)
    data = data.sort_values(
        ["_model_rank", "concurrency", "_workload_rank", "_mechanism_rank"]
    ).drop(columns=["_model_rank", "_workload_rank", "_mechanism_rank"])
    return data[CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.8f", na_rep="")
    serialized = pd.read_csv(path, keep_default_na=False)
    require(list(serialized.columns) == CSV_COLUMNS, "serialized CSV columns differ")
    require(len(serialized) == 108, "serialized CSV row count differs")
    for column in (
        "e2e_ms",
        "adaptmoe_e2e_ms",
        "normalized_e2e",
        "offload_gib",
        "geomean_normalized_e2e",
        "panel_max_valid_ratio",
    ):
        serialized[column] = pd.to_numeric(serialized[column], errors="coerce")
    for column in ("is_ratio_below_one", "is_panel_slowest_valid_baseline"):
        if not pd.api.types.is_bool_dtype(serialized[column]):
            serialized[column] = serialized[column].map({"True": True, "False": False})
        require(serialized[column].notna().all(), f"serialized boolean differs: {column}")
    require(serialized.loc[serialized["status"] == "OOM", "e2e_ms"].isna().all(), "OOM E2E exists")
    require(
        serialized.loc[serialized["status"] == "OOM", "normalized_e2e"].isna().all(),
        "OOM ratio exists",
    )
    return serialized


def panel_data(data: pd.DataFrame, model: str, concurrency: int) -> pd.DataFrame:
    selected = data[(data["model"] == model) & (data["concurrency"] == concurrency)]
    expected = len(MODEL_WORKLOADS[model]) * len(MECHANISMS)
    require(len(selected) == expected, f"panel row count differs: {model}/C{concurrency}")
    return selected


def lookup(panel: pd.DataFrame, workload: str, mechanism: str) -> pd.Series:
    selected = panel[(panel["workload"] == workload) & (panel["mechanism"] == mechanism)]
    require(len(selected) == 1, f"point lookup differs: {workload}/{mechanism}")
    return selected.iloc[0]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.3,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
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


def ratio_tick(value: float, _position: int) -> str:
    return f"{value:g}×"


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    valid_ratios = data.loc[data["status"] == "ok", "normalized_e2e"].to_numpy(dtype=float)
    maximum = float(valid_ratios.max())
    ticks = [0.5, 1.0, 2.0, 4.0, 8.0]
    if maximum > 8.0:
        ticks.append(16.0)
    x_min = 0.45
    x_max = 17.5 if maximum > 8.0 else 8.8
    oom_x = 15.2 if maximum > 8.0 else 7.65

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
        current = panel_data(data, model, concurrency)
        workloads = MODEL_WORKLOADS[model]
        y_base = np.arange(len(workloads), dtype=float)
        for y in y_base:
            axis.axhline(y, color="#E2E2E2", linewidth=0.55, zorder=0)
        axis.axvline(1.0, color="#6F8291", linewidth=1.15, zorder=1)

        for workload_index, workload in enumerate(workloads):
            for mechanism in MECHANISMS:
                row = lookup(current, workload, mechanism)
                y = workload_index + Y_OFFSETS[mechanism]
                style = STYLES[mechanism]
                if row["status"] == "OOM":
                    axis.scatter(
                        [oom_x],
                        [y],
                        s=style["size"] ** 2,
                        marker="s",
                        facecolors=style["color"],
                        edgecolors=style["color"],
                        linewidths=1.0,
                        zorder=4,
                    )
                    axis.scatter(
                        [oom_x],
                        [y],
                        s=(style["size"] + 1.2) ** 2,
                        marker="x",
                        color="#222222",
                        linewidths=1.0,
                        zorder=5,
                    )
                    axis.annotate(
                        "OOM",
                        (oom_x, y),
                        xytext=(-4, 0),
                        textcoords="offset points",
                        ha="right",
                        va="center",
                        fontsize=6.5,
                        fontweight="bold",
                        color="#333333",
                    )
                    continue

                ratio = float(row["normalized_e2e"])
                if mechanism != "AdaptMoE":
                    axis.plot(
                        [min(1.0, ratio), max(1.0, ratio)],
                        [y, y],
                        color="#CFCFCF",
                        linewidth=0.65,
                        solid_capstyle="round",
                        zorder=1,
                    )
                facecolor = style["color"] if style["filled"] else "white"
                axis.scatter(
                    [ratio],
                    [y],
                    s=style["size"] ** 2,
                    marker=style["marker"],
                    facecolors=facecolor,
                    edgecolors=style["color"],
                    linewidths=1.15,
                    zorder=5 if mechanism == "AdaptMoE" else 3,
                )
                should_label = bool(row["is_ratio_below_one"]) or bool(
                    row["is_panel_slowest_valid_baseline"]
                )
                if should_label:
                    is_slowest = bool(row["is_panel_slowest_valid_baseline"])
                    axis.annotate(
                        f"{ratio:.2f}×",
                        (ratio, y),
                        xytext=(-3 if is_slowest else 3, -3.5),
                        textcoords="offset points",
                        ha="right" if is_slowest else "left",
                        va="top",
                        fontsize=6.5,
                        color=style["color"],
                        fontweight="bold",
                        zorder=6,
                    )

        axis.set_xscale("log", base=2)
        axis.set_xlim(x_min, x_max)
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_major_formatter(FuncFormatter(ratio_tick))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_ylim(len(workloads) - 0.45, -0.90)
        axis.set_yticks(y_base)
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
            0.94,
            0.965,
            "Faster than\nAdaptMoE",
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=5.7,
            color="#687884",
            linespacing=0.9,
        )
        axis.text(
            1.07,
            0.965,
            "Slower than\nAdaptMoE",
            transform=axis.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=5.7,
            color="#687884",
            linespacing=0.9,
        )
        axis.text(
            0.985,
            0.965,
            "Lower is better",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.0,
            fontstyle="italic",
            color="#555555",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    legend_order = (
        "AdaptMoE",
        "KTransformers",
        "Native UVA",
        "UVA Dual-Graph",
        "Native Copy",
        "Copy Dual-Graph",
    )
    handles = []
    for mechanism in legend_order:
        style = STYLES[mechanism]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=style["marker"],
                markersize=style["size"],
                markerfacecolor=style["color"] if style["filled"] else "white",
                markeredgecolor=style["color"],
                markeredgewidth=1.15,
                label=mechanism,
            )
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.3,
    )
    figure.subplots_adjust(left=0.11, right=0.985, bottom=0.13, top=0.84, wspace=0.18, hspace=0.31)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "End-to-end latency normalized to AdaptMoE",
        "Subject": "Six execution configurations across two models and two concurrencies",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(png_path, format="png", dpi=300, metadata={"Software": SCRIPT_PATH.name})
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "End-to-end latency normalized to AdaptMoE for six execution configurations. "
        "Each point reports the latency of one configuration divided by AdaptMoE under "
        "the same model, workload, and concurrency. The vertical line at 1× denotes "
        "AdaptMoE; points to the right are slower and points to the left are faster. "
        "Hollow and filled markers distinguish baseline and dual-Graph variants of the "
        "same data path. OOM denotes configurations that cannot complete because of "
        "Graph-pool or temporary-workspace capacity and is not numericized. Values are "
        "single-run measurements assembled by the latest consolidated matrix; newly "
        "measured cells use independent cold processes, and no repeat-based error bars "
        "are shown. Offload placements are reported in the CSV and are not identical "
        "for every mechanism."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_summary(data: pd.DataFrame) -> None:
    for model in MODEL_WORKLOADS:
        for concurrency in CONCURRENCIES:
            current = panel_data(data, model, concurrency)
            print(f"[{model}, C{concurrency}]")
            for mechanism in MECHANISMS[1:]:
                row = current[current["mechanism"] == mechanism].iloc[0]
                valid_count = int(
                    ((current["mechanism"] == mechanism) & (current["status"] == "ok")).sum()
                )
                print(
                    f"  {mechanism} geometric mean: "
                    f"{float(row['geomean_normalized_e2e']):.4f}× (n={valid_count})"
                )
            lowest_count = int(current["adaptmoe_lowest_workload_count"].iloc[0])
            print(f"  AdaptMoE lowest-E2E workloads: {lowest_count}/{len(MODEL_WORKLOADS[model])}")
            below = current[current["is_ratio_below_one"]]
            below_text = ", ".join(
                f"{row.workload}/{row.mechanism}={row.normalized_e2e:.4f}×"
                for row in below.itertuples()
            ) or "none"
            print(f"  Baseline ratios < 1: {below_text}")
            print(f"  Maximum valid baseline ratio: {current['panel_max_valid_ratio'].iloc[0]:.4f}×")
            oom = current[current["status"] == "OOM"]
            oom_text = ", ".join(f"{row.workload}/{row.mechanism}" for row in oom.itertuples()) or "none"
            print(f"  OOM: {oom_text}")


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    csv_path = args.csv.resolve()
    pdf_path = args.pdf.resolve()
    png_path = args.png.resolve()
    caption_path = args.caption.resolve()
    require(report_path.is_file(), f"report not found: {report_path}")

    data = build_data(report_path)
    plot_data = write_and_reload_csv(data, csv_path)
    render_figure(plot_data, pdf_path, png_path)
    write_caption(caption_path)

    print(f"[OK] source: {report_path}")
    print(f"[OK] CSV: {csv_path} ({len(plot_data)} rows)")
    print(f"[OK] PDF: {pdf_path}")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] caption: {caption_path}")
    print_summary(plot_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
