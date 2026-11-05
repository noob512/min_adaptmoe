#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Compare Stock UVM demand paging with the requested Native UVA baselines."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_stock_uvm_vs_native_uva"

DEFAULT_STOCK_REPORT = REPO_ROOT / "evaluation/data/processed/REPORT.md"
DEFAULT_NATIVE_REPORT = (
    REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E14_CONCISE_Full.md"
)
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

STOCK_SECTION_START = "### 11.3 vLLM Device-resident 与 Stock UVM Demand 单轮对照"
STOCK_SECTION_END = "### 11.4 H 的结论和限制"

MODELS = (
    "DeepSeek-V2-Lite BF16",
    "Qwen3-30B-A3B FP8",
)
MODEL_BY_STOCK_SUBSECTION = {
    "11.3.1": MODELS[0],
    "11.3.2": MODELS[1],
}
NATIVE_MECHANISM = {
    MODELS[0]: "Native UVA（eager Prefill + Decode Graph）",
    MODELS[1]: "Native UVA eager",
}
CONFIGS = {
    MODELS[0]: [
        ("Short", 1),
        ("Short", 16),
        ("Ordinary", 1),
        ("Ordinary", 16),
        ("RAG", 1),
        ("RAG", 16),
        ("Long", 1),
        ("Long", 16),
    ],
    MODELS[1]: [
        ("Short", 1),
        ("Short", 16),
        ("Ordinary", 1),
        ("Ordinary", 16),
        ("Medium", 1),
        ("Medium", 16),
        ("RAG", 1),
        ("RAG", 16),
        ("Long", 1),
        ("Long", 16),
    ],
}
METRICS = ("TTFT", "TPOT", "E2E")

CSV_COLUMNS = [
    "model",
    "workload",
    "concurrency",
    "metric",
    "stock_uvm_ms",
    "native_uva_ms",
    "ratio",
    "log2_ratio",
    "source_section",
    "validity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-report", type=Path, default=DEFAULT_STOCK_REPORT)
    parser.add_argument("--native-report", type=Path, default=DEFAULT_NATIVE_REPORT)
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


def parse_markdown_table(block: str, label: str) -> pd.DataFrame:
    table_lines = [line for line in block.splitlines() if line.startswith("|")]
    require(len(table_lines) >= 3, f"no Markdown table found in {label}")
    header = split_markdown_row(table_lines[0])
    rows = [
        split_markdown_row(line)
        for line in table_lines[2:]
        if not re.fullmatch(r"[|:\- ]+", line)
    ]
    require(rows and all(len(row) == len(header) for row in rows), f"bad table in {label}")
    return pd.DataFrame(rows, columns=header)


def parse_ms(value: object, *, label: str) -> float:
    text = str(value).strip()
    require(text not in {"OOM", "—", "N/A", ""}, f"invalid value in {label}: {text}")
    result = float(text.replace(",", ""))
    require(math.isfinite(result) and result > 0, f"non-positive value in {label}")
    return result


def extract_stock_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    start = report.find(STOCK_SECTION_START)
    end = report.find(STOCK_SECTION_END, start)
    require(start >= 0 and end > start, "Stock report §11.3 boundaries not found")
    section = report[start:end]
    require("64 MiB Stock UVM 微基准" not in section, "§11.1 leaked into Stock input")

    headings = list(re.finditer(r"^#### (11\.3\.[12]) .+$", section, re.MULTILINE))
    require(len(headings) == 2, "expected Stock subsections §11.3.1 and §11.3.2")
    records: list[dict[str, object]] = []
    for index, heading in enumerate(headings):
        subsection = heading.group(1)
        model = MODEL_BY_STOCK_SUBSECTION[subsection]
        block_end = headings[index + 1].start() if index + 1 < len(headings) else len(section)
        table = parse_markdown_table(section[heading.end() : block_end], f"§{subsection}")
        expected_columns = {
            "请求",
            "C",
            "UVM TTFT",
            "UVM TPOT",
            "UVM E2E",
        }
        require(expected_columns.issubset(table.columns), f"Stock columns differ in §{subsection}")

        by_config = {
            (str(row["请求"]), int(row["C"])): row for _, row in table.iterrows()
        }
        require(
            set(by_config) == set(CONFIGS[model]),
            f"Stock configuration coverage differs for {model}",
        )
        for workload, concurrency in CONFIGS[model]:
            row = by_config[(workload, concurrency)]
            for metric in METRICS:
                records.append(
                    {
                        "model": model,
                        "workload": workload,
                        "concurrency": concurrency,
                        "metric": metric,
                        "stock_uvm_ms": parse_ms(
                            row[f"UVM {metric}"],
                            label=f"Stock {model}/{workload}/C{concurrency}/{metric}",
                        ),
                        "stock_section": f"REPORT.md §{subsection}",
                    }
                )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 54, f"expected 54 Stock observations, found {len(data)}")
    return data


def native_model_block(report: str, model: str) -> str:
    start_marker = f"## {model}"
    start = report.find(start_marker)
    require(start >= 0, f"Native report section not found: {model}")
    end = report.find("\n## ", start + len(start_marker))
    if end < 0:
        end = len(report)
    return report[start:end]


def extract_native_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    records: list[dict[str, object]] = []
    for model in MODELS:
        block = native_model_block(report, model)
        table = parse_markdown_table(block, f"Native {model}")
        expected_columns = {"场景", "C", "机制", "TTFT (ms)", "TPOT (ms)", "E2E (ms)", "状态"}
        require(expected_columns.issubset(table.columns), f"Native columns differ for {model}")
        mechanism = NATIVE_MECHANISM[model]
        selected = table[table["机制"] == mechanism].copy()
        require(
            len(selected) == len(CONFIGS[model]),
            f"expected {len(CONFIGS[model])} exact Native UVA rows for {model}, found {len(selected)}",
        )
        require(
            selected["状态"].str.startswith("ok").all(),
            f"a selected Native UVA row is not valid for {model}",
        )
        selected["config"] = list(zip(selected["场景"], selected["C"].astype(int)))
        by_config = {row["config"]: row for _, row in selected.iterrows()}
        require(
            set(by_config) == set(CONFIGS[model]),
            f"Native UVA configuration coverage differs for {model}",
        )

        for workload, concurrency in CONFIGS[model]:
            row = by_config[(workload, concurrency)]
            for metric in METRICS:
                records.append(
                    {
                        "model": model,
                        "workload": workload,
                        "concurrency": concurrency,
                        "metric": metric,
                        "native_uva_ms": parse_ms(
                            row[f"{metric} (ms)"],
                            label=f"Native {model}/{workload}/C{concurrency}/{metric}",
                        ),
                        "native_section": (
                            f"RESULTS_E0_E14_CONCISE_Full.md §{model}; mechanism={mechanism}"
                        ),
                    }
                )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 54, f"expected 54 Native observations, found {len(data)}")
    return data


def merge_sources(stock: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "workload", "concurrency", "metric"]
    merged = stock.merge(native, on=keys, how="outer", validate="one_to_one", indicator=True)
    require((merged["_merge"] == "both").all(), "Stock and Native configuration sets differ")
    merged = merged.drop(columns="_merge")
    merged["ratio"] = merged["stock_uvm_ms"] / merged["native_uva_ms"]
    merged["log2_ratio"] = np.log2(merged["ratio"])
    merged["source_section"] = merged["stock_section"] + " + " + merged["native_section"]
    merged["validity"] = "valid_completed_single_run_unpaired"
    merged = merged.drop(columns=["stock_section", "native_section"])

    order_records = [
        (model, workload, concurrency, metric)
        for model in MODELS
        for workload, concurrency in CONFIGS[model]
        for metric in METRICS
    ]
    order = pd.MultiIndex.from_tuples(order_records, names=keys)
    merged = merged.set_index(keys).loc[order].reset_index()
    require(len(merged) == 54, f"expected 54 merged observations, found {len(merged)}")
    require(np.isfinite(merged["ratio"]).all() and (merged["ratio"] > 0).all(), "bad ratios")
    return merged[CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.8f")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == CSV_COLUMNS, "serialized CSV columns differ")
    require(len(serialized) == 54, "serialized CSV row count differs")
    require(
        set(serialized["validity"]) == {"valid_completed_single_run_unpaired"},
        "unexpected validity value",
    )
    require(
        np.allclose(
            serialized["ratio"],
            serialized["stock_uvm_ms"] / serialized["native_uva_ms"],
            rtol=0,
            atol=5e-9,
        ),
        "serialized ratios do not match source milliseconds",
    )
    require(
        np.allclose(
            serialized["log2_ratio"],
            np.log2(serialized["ratio"]),
            rtol=0,
            atol=5e-8,
        ),
        "serialized log2 ratios differ",
    )
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.3,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def heatmap_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "uvm_ratio_diverging",
        ["#0072B2", "#B8D9EA", "#F0F0F0", "#F4B45F", "#D55E00"],
        N=256,
    )


def text_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _alpha = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance >= 0.55 else "white"


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    cmap = heatmap_colormap()
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=3.0)

    fig = plt.figure(figsize=(7.0, 3.65))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 0.035],
        height_ratios=[1.0, 1.12],
        left=0.105,
        right=0.885,
        bottom=0.145,
        top=0.955,
        hspace=0.57,
        wspace=0.07,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    colorbar_axis = fig.add_subplot(grid[:, 1])
    panel_titles = {
        MODELS[0]: "(a) DeepSeek-V2-Lite BF16",
        MODELS[1]: "(b) Qwen3-30B-A3B FP8",
    }

    image = None
    for ax, model in zip(axes, MODELS, strict=True):
        configs = CONFIGS[model]
        panel = data[data["model"] == model]
        values = np.empty((len(METRICS), len(configs)), dtype=float)
        ratios = np.empty_like(values)
        for row_index, metric in enumerate(METRICS):
            lookup = panel[panel["metric"] == metric].set_index(["workload", "concurrency"])
            for column_index, config in enumerate(configs):
                ratios[row_index, column_index] = float(lookup.loc[config, "ratio"])
                values[row_index, column_index] = float(lookup.loc[config, "log2_ratio"])

        image = ax.pcolormesh(
            np.arange(len(configs) + 1),
            np.arange(len(METRICS) + 1),
            values,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="white",
            linewidth=0.65,
            antialiased=True,
            rasterized=False,
        )
        for row_index, metric in enumerate(METRICS):
            for column_index in range(len(configs)):
                rgba = cmap(norm(values[row_index, column_index]))
                ax.text(
                    column_index + 0.5,
                    row_index + 0.5,
                    f"{ratios[row_index, column_index]:.2f}×",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold" if metric == "E2E" else "normal",
                    color=text_color(rgba),
                )

        labels = [f"{workload}\nC{concurrency}" for workload, concurrency in configs]
        ax.set_xticks(np.arange(len(configs)) + 0.5, labels=labels)
        ax.set_yticks(np.arange(len(METRICS)) + 0.5, labels=METRICS)
        ax.invert_yaxis()
        ax.set_xlim(0, len(configs))
        ax.set_ylim(len(METRICS), 0)
        ax.set_xlabel("Workload / Concurrency", labelpad=3.0)
        ax.set_title(panel_titles[model], loc="left", pad=4.0)
        ax.axhline(2.0, color="white", linewidth=1.35, zorder=3)
        for tick, metric in zip(ax.get_yticklabels(), METRICS, strict=True):
            if metric == "E2E":
                tick.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_visible(False)

    require(image is not None, "heatmap was not rendered")
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    # Matplotlib rasterizes long colorbar meshes by default; keep the complete
    # PDF vector-native for paper production.
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    ratio_ticks = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    colorbar.set_ticks(np.log2(ratio_ticks))
    colorbar.set_ticklabels(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
    colorbar.ax.tick_params(labelsize=7.0, width=0.7, length=2.5)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("Stock UVM / Native UVA", fontsize=8.0, labelpad=7.0)
    colorbar.ax.text(
        0.5,
        1.02,
        "UVM slower",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Stock UVM versus Native UVA latency",
        "Subject": "Annotated heatmaps of Stock UVM / Native UVA latency ratios",
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


def write_caption(path: Path) -> None:
    caption = (
        "Latency of Stock UVM demand paging normalized to Native UVA remote access. "
        "Each cell reports Stock UVM / Native UVA for request-level P50 latency in "
        "milliseconds, where values above 1× indicate that demand paging is slower. "
        "Panels cover DeepSeek-V2-Lite BF16 and Qwen3-30B-A3B FP8 under Short, "
        "Ordinary, RAG, Long, and (for Qwen) Medium workloads at concurrency C1 and "
        "C16. The DeepSeek Native UVA baseline uses eager Prefill with a Decode Graph; "
        "the Qwen baseline is eager in both phases. Each source cell is one cold-process "
        "result and the runs are unpaired, so no error bars or confidence intervals are "
        "shown. The comparison is limited to one RTX 5090, frozen requests, and the "
        "reported cross-run protocols. Stock UVM increases end-to-end latency across "
        "the evaluated workloads, showing that automatic page migration introduces "
        "substantial serving overhead compared with directly accessing pinned host "
        "memory."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_statistics(data: pd.DataFrame) -> None:
    for metric in METRICS:
        values = data.loc[data["metric"] == metric, "ratio"]
        print(f"{metric} ratio min/max: {values.min():.4f}× / {values.max():.4f}×")
    e2e = data[data["metric"] == "E2E"]
    print(f"E2E ratio > 1 cells: {int((e2e['ratio'] > 1.0).sum())}/{len(e2e)}")
    for model in MODELS:
        values = e2e.loc[e2e["model"] == model, "ratio"].to_numpy(dtype=float)
        geometric_mean = float(np.exp(np.log(values).mean()))
        print(f"{model} E2E geometric mean ratio: {geometric_mean:.4f}×")


def main() -> int:
    args = parse_args()
    stock_report = args.stock_report.resolve()
    native_report = args.native_report.resolve()
    csv_path = args.csv.resolve()
    pdf_path = args.pdf.resolve()
    png_path = args.png.resolve()
    caption_path = args.caption.resolve()
    require(stock_report.is_file(), f"Stock report not found: {stock_report}")
    require(native_report.is_file(), f"Native report not found: {native_report}")

    stock = extract_stock_data(stock_report)
    native = extract_native_data(native_report)
    merged = merge_sources(stock, native)
    plot_data = write_and_reload_csv(merged, csv_path)
    render_figure(plot_data, pdf_path, png_path)
    write_caption(caption_path)

    print(f"[OK] CSV: {csv_path} ({len(plot_data)} rows)")
    print(f"[OK] PDF: {pdf_path}")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] caption: {caption_path}")
    print_statistics(plot_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
