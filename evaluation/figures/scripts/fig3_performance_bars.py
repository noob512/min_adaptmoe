#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Generate Figure 3 performance bars from the E0--E14 Markdown matrix."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_REPORT = REPO_ROOT / "evaluation/data/processed/RESULTS_E0_E14_CONCISE_Full.md"

MODELS = {
    "DeepSeek-V2-Lite BF16": {
        "stem": "fig3a_deepseek_performance",
        "workloads": ("Short", "Ordinary", "RAG", "Long"),
        "panel": "a",
    },
    "Qwen3-30B-A3B FP8": {
        "stem": "fig3b_qwen_performance",
        "workloads": ("Short", "Ordinary", "Medium", "RAG", "Long"),
        "panel": "b",
    },
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
        "Native UVA 双阶段 Graph": "UVA Dual-Graph",
        "Native Copy eager": "Native Copy",
        "Native Copy 双阶段 Graph": "Copy Dual-Graph",
        "KTransformers 7T + Decode Graph": "KTransformers",
    },
    "Qwen3-30B-A3B FP8": {
        "AdaptMoE 双阶段 Graph": "AdaptMoE",
        "Native UVA eager": "Native UVA",
        "Native UVA 双阶段 Graph": "UVA Dual-Graph",
        "Native Copy eager": "Native Copy",
        "Native Copy 双阶段 Graph": "Copy Dual-Graph",
        "KTransformers 7T + Decode Graph": "KTransformers",
    },
}

STYLES = {
    "AdaptMoE": {"face": "#0072B2", "hatch": "//"},
    "Native UVA": {"face": "#F6C85F", "hatch": ".."},
    "UVA Dual-Graph": {"face": "#E69F00", "hatch": None},
    "Native Copy": {"face": "#A8DDB5", "hatch": "\\\\"},
    "Copy Dual-Graph": {"face": "#009E73", "hatch": None},
    "KTransformers": {"face": "#7E57C2", "hatch": "xx"},
}
EDGE_COLOR = "#343434"
GROUP_WIDTH = 0.84
BAR_WIDTH = 0.13

METRICS = (
    {
        "column": "decode_tokens_per_s",
        "label": "Decode throughput (tokens/s)",
        "direction": "Higher is better",
        "scale": "linear",
    },
    {
        "column": "ttft_s",
        "label": "TTFT (s)",
        "direction": "Lower is better",
        "scale": "log",
    },
    {
        "column": "e2e_s",
        "label": "End-to-end latency (s)",
        "direction": "Lower is better",
        "scale": "log",
    },
    {
        "column": "tpot_ms",
        "label": "TPOT (ms)",
        "direction": "Lower is better",
        "scale": "log",
    },
)

CSV_COLUMNS = (
    "model",
    "workload",
    "concurrency",
    "mechanism",
    "offload_gib",
    "ttft_ms",
    "ttft_s",
    "tpot_ms",
    "e2e_ms",
    "e2e_s",
    "decode_tokens_per_s",
    "status",
    "source_section",
    "validity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-root", type=Path, default=FIGURES_ROOT)
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


def parse_primary_table(block: str, model: str) -> pd.DataFrame:
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
    lines = block.splitlines()
    start = next((i for i, line in enumerate(lines) if split_markdown_row(line) == header), -1)
    require(start >= 0, f"primary table header missing: {model}")
    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    expected = len(MODELS[model]["workloads"]) * len(CONCURRENCIES) * len(MECHANISMS)
    require(len(table_lines) == expected + 2, f"primary table row count differs: {model}")
    rows = [split_markdown_row(line) for line in table_lines[2:]]
    require(all(len(row) == len(header) for row in rows), f"malformed table row: {model}")
    return pd.DataFrame(rows, columns=header)


def parse_positive(cell: str, label: str) -> float:
    cleaned = cell.strip().replace(",", "")
    require(re.fullmatch(r"\d+(?:\.\d+)?", cleaned) is not None, f"invalid {label}: {cell}")
    value = float(cleaned)
    require(value > 0 and math.isfinite(value), f"non-positive {label}: {cell}")
    return value


def build_data(report_path: Path) -> pd.DataFrame:
    report = report_path.read_text(encoding="utf-8")
    require("# 两模型 C1/C16 六机制完整结果" in report, "unexpected source document")
    blocks = {
        "DeepSeek-V2-Lite BF16": extract_model_block(
            report, "DeepSeek-V2-Lite BF16", "## Qwen3-30B-A3B FP8"
        ),
        "Qwen3-30B-A3B FP8": extract_model_block(
            report, "Qwen3-30B-A3B FP8", "## 覆盖与失败"
        ),
    }

    records: list[dict[str, object]] = []
    metric_source_columns = {
        "ttft_ms": "TTFT (ms)",
        "tpot_ms": "TPOT (ms)",
        "e2e_ms": "E2E (ms)",
        "decode_tokens_per_s": "Decode tok/s",
    }
    for model, metadata in MODELS.items():
        table = parse_primary_table(blocks[model], model)
        mapping = SOURCE_MECHANISMS[model]
        require(set(table["机制"]) == set(mapping), f"mechanism set differs: {model}")
        for _, row in table.iterrows():
            workload = str(row["场景"])
            concurrency = int(row["C"])
            require(workload in metadata["workloads"], f"unexpected workload: {model}/{workload}")
            require(concurrency in CONCURRENCIES, f"unexpected concurrency: {concurrency}")
            status_text = str(row["状态"])
            if status_text.startswith("ok"):
                status = "ok"
                validity = "valid_single_sample"
                metrics = {
                    output: parse_positive(str(row[source_column]), source_column)
                    for output, source_column in metric_source_columns.items()
                }
            elif status_text.startswith("OOM"):
                status = "OOM"
                validity = "oom_not_numeric"
                require(
                    all(str(row[column]) == "—" for column in metric_source_columns.values()),
                    f"OOM row contains numeric metrics: {model}/{workload}/C{concurrency}",
                )
                metrics = {output: math.nan for output in metric_source_columns}
            else:
                status = "missing"
                validity = "missing_not_numeric"
                require(
                    all(str(row[column]) == "—" for column in metric_source_columns.values()),
                    f"missing row contains numeric metrics: {model}/{workload}/C{concurrency}",
                )
                metrics = {output: math.nan for output in metric_source_columns}

            records.append(
                {
                    "model": model,
                    "workload": workload,
                    "concurrency": concurrency,
                    "mechanism": mapping[str(row["机制"])],
                    "offload_gib": parse_positive(str(row["实际卸载 GiB"]), "offload GiB"),
                    **metrics,
                    "status": status,
                    "source_section": f"RESULTS_E0_E14_CONCISE_Full.md §{model} primary table",
                    "validity": validity,
                }
            )

    data = pd.DataFrame.from_records(records)
    require(len(data) == 108, "expected 108 primary-table cells")
    data["ttft_s"] = data["ttft_ms"] / 1000.0
    data["e2e_s"] = data["e2e_ms"] / 1000.0
    for column in ("ttft_s", "e2e_s"):
        data.loc[data["status"] != "ok", column] = math.nan

    model_rank = {model: i for i, model in enumerate(MODELS)}
    workload_rank = {
        (model, workload): i
        for model, metadata in MODELS.items()
        for i, workload in enumerate(metadata["workloads"])
    }
    mechanism_rank = {mechanism: i for i, mechanism in enumerate(MECHANISMS)}
    data["_model"] = data["model"].map(model_rank)
    data["_workload"] = [
        workload_rank[(model, workload)]
        for model, workload in zip(data["model"], data["workload"])
    ]
    data["_mechanism"] = data["mechanism"].map(mechanism_rank)
    data = data.sort_values(
        ["_model", "concurrency", "_workload", "_mechanism"]
    ).drop(columns=["_model", "_workload", "_mechanism"])
    require(not data.duplicated(["model", "workload", "concurrency", "mechanism"]).any(),
            "duplicate cells")
    return data.loc[:, CSV_COLUMNS]


def write_model_csv(data: pd.DataFrame, model: str, path: Path) -> pd.DataFrame:
    model_data = data[data["model"] == model].copy()
    expected = len(MODELS[model]["workloads"]) * len(CONCURRENCIES) * len(MECHANISMS)
    require(len(model_data) == expected, f"model row count differs: {model}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(path, index=False, float_format="%.8f", na_rep="")
    serialized = pd.read_csv(path, keep_default_na=False)
    require(list(serialized.columns) == list(CSV_COLUMNS), f"CSV columns differ: {path}")
    require(len(serialized) == expected, f"serialized row count differs: {path}")
    for column in CSV_COLUMNS[4:11]:
        serialized[column] = pd.to_numeric(serialized[column], errors="coerce")
    invalid = serialized["status"] != "ok"
    require(serialized.loc[invalid, list(CSV_COLUMNS[5:11])].isna().all().all(),
            "invalid metrics became numeric")
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.3,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def lookup(panel: pd.DataFrame, workload: str, mechanism: str) -> pd.Series:
    row = panel[(panel["workload"] == workload) & (panel["mechanism"] == mechanism)]
    require(len(row) == 1, f"cell lookup differs: {workload}/{mechanism}")
    return row.iloc[0]


def format_value(metric: str, value: float) -> str:
    if metric == "decode_tokens_per_s":
        return f"{value:.0f}"
    if metric == "ttft_s":
        return f"{value:.2f}" if value < 1.0 else f"{value:.1f}"
    if metric == "e2e_s":
        return f"{value:.2f}" if value < 10.0 else f"{value:.1f}"
    if metric == "tpot_ms":
        return f"{value:.1f}"
    raise ValueError(f"unknown metric: {metric}")


def log_oom_y(lower: float, upper: float) -> float:
    log_lower, log_upper = math.log10(lower), math.log10(upper)
    return 10 ** (log_upper - 0.07 * (log_upper - log_lower))


def render_model(data: pd.DataFrame, model: str, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    metadata = MODELS[model]
    workloads = metadata["workloads"]
    centers = np.arange(len(workloads), dtype=float)
    offsets = {
        mechanism: (i - (len(MECHANISMS) - 1) / 2.0) * BAR_WIDTH
        for i, mechanism in enumerate(MECHANISMS)
    }
    # Fill the page row-first: one row per concurrency and one column per
    # metric.  This keeps the four related metrics adjacent instead of
    # stretching a metric pair across four stacked rows.
    figure, axes = plt.subplots(2, 4, figsize=(7.0, 4.35), sharex="col")

    for row_index, concurrency in enumerate(CONCURRENCIES):
        for column_index, metric in enumerate(METRICS):
            axis = axes[row_index, column_index]
            panel = data[data["concurrency"] == concurrency]
            valid_values = panel.loc[panel["status"] == "ok", metric["column"]].astype(float)
            require(len(valid_values) > 0, f"no valid values: {model}/C{concurrency}/{metric['column']}")

            if metric["scale"] == "linear":
                lower, upper = 0.0, float(valid_values.max()) * 1.27
                axis.set_ylim(lower, upper)
            else:
                raw_lower = float(valid_values.min())
                raw_upper = float(valid_values.max())
                lower = 10 ** (math.log10(raw_lower) - 0.18)
                upper = 10 ** (math.log10(raw_upper) + 0.25)
                axis.set_yscale("log")
                axis.set_ylim(lower, upper)
                axis.yaxis.set_major_locator(LogLocator(base=10))
                axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
                axis.yaxis.set_minor_formatter(NullFormatter())

            bar_records: list[tuple[str, str, float, float, object]] = []
            for workload_index, workload in enumerate(workloads):
                for mechanism in MECHANISMS:
                    source_row = lookup(panel, workload, mechanism)
                    x = workload_index + offsets[mechanism]
                    style = STYLES[mechanism]
                    if source_row["status"] != "ok":
                        oom_y = upper * 0.91 if metric["scale"] == "linear" else log_oom_y(lower, upper)
                        axis.scatter(
                            [x], [oom_y], marker="x", s=25, color=EDGE_COLOR,
                            linewidths=1.0, zorder=6,
                        )
                        axis.annotate(
                            "OOM" if source_row["status"] == "OOM" else "N/A",
                            (x, oom_y), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=5.7,
                            fontweight="bold", clip_on=False,
                        )
                        continue
                    value = float(source_row[metric["column"]])
                    axis.bar(
                        x, value, width=BAR_WIDTH,
                        color=style["face"], edgecolor=EDGE_COLOR,
                        linewidth=0.55, hatch=style["hatch"], zorder=3,
                    )
                    bar_records.append((workload, mechanism, x, value, source_row))

            # Keep numeric labels only on AdaptMoE bars to avoid visual clutter.
            for workload, mechanism, x, value, _source_row in bar_records:
                if mechanism != "AdaptMoE":
                    continue
                axis.annotate(
                    format_value(metric["column"], value),
                    (x, value),
                    xytext=(0, 2.5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5.7,
                    fontweight="bold",
                    color="#005E93",
                    clip_on=False,
                    zorder=7,
                )

            axis.set_xlim(-0.58, len(workloads) - 0.42)
            axis.set_axisbelow(True)
            axis.grid(axis="y", which="major", color="#DCDCDC", linewidth=0.55)
            axis.grid(axis="x", visible=False)
            axis.set_xticks(centers)
            if row_index == len(CONCURRENCIES) - 1:
                rotation = 42 if model.startswith("Qwen") else 32
                axis.set_xticklabels(
                    workloads,
                    rotation=rotation,
                    ha="right",
                    rotation_mode="anchor",
                    fontsize=6.3,
                )
            else:
                axis.tick_params(axis="x", labelbottom=False)

            if row_index == 0:
                title = metric["label"].replace("Decode throughput ", "Decode throughput\n")
                title = title.replace("End-to-end latency ", "End-to-end latency\n")
                axis.set_title(title, pad=3.0, fontweight="bold")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    # Matplotlib fills a multi-row legend column-wise. This ordering displays
    # the mechanisms left-to-right in the same order used within each group.
    legend_order = (
        "AdaptMoE", "Native Copy", "Native UVA",
        "Copy Dual-Graph", "UVA Dual-Graph", "KTransformers",
    )
    handles = [
        Patch(
            facecolor=STYLES[mechanism]["face"], edgecolor=EDGE_COLOR,
            linewidth=0.6, hatch=STYLES[mechanism]["hatch"], label=mechanism,
        )
        for mechanism in legend_order
    ]
    figure.suptitle(model, y=0.995, fontsize=9.0, fontweight="bold")
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.947),
        ncol=3, frameon=False, handlelength=1.5, handleheight=0.9,
        columnspacing=1.2, labelspacing=0.5,
    )
    figure.text(0.018, 0.615, "B=1", rotation=90, ha="center", va="center",
                fontsize=8.0, fontweight="bold")
    figure.text(0.018, 0.265, "B=16", rotation=90, ha="center", va="center",
                fontsize=8.0, fontweight="bold")
    figure.supxlabel("Workload", x=0.535, y=0.018, fontsize=8.2)
    figure.subplots_adjust(
        left=0.075, right=0.992, bottom=0.17, top=0.80,
        wspace=0.34, hspace=0.31,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    save_metadata = {
        "Title": f"Performance comparison on {model}",
        "Subject": "Decode throughput, TTFT, end-to-end latency, and TPOT",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=save_metadata)
    figure.savefig(
        png_path, format="png", dpi=300,
        metadata={"Software": SCRIPT_PATH.name},
    )
    plt.close(figure)


def write_caption(model: str, path: Path) -> None:
    if model.startswith("DeepSeek"):
        caption = (
            "Performance comparison of six execution configurations on "
            "DeepSeek-V2-Lite BF16 at batch sizes 1 and 16. Rows report steady "
            "decode throughput, TTFT, end-to-end latency, and TPOT. Higher throughput "
            "and lower latency are better. Values are single-run observations; no "
            "repeat-based error bars are shown."
        )
    else:
        caption = (
            "Performance comparison of six execution configurations on "
            "Qwen3-30B-A3B FP8 at batch sizes 1 and 16. AdaptMoE combines phase-specific "
            "CUDA Graphs with GPU-local/copy prefill and remote decode. Rows report "
            "steady decode throughput, TTFT, end-to-end latency, and TPOT. Values are "
            "single-run observations; no repeat-based error bars are shown."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    output_root = args.output_root.resolve()
    require(report_path.is_file(), f"report not found: {report_path}")
    data = build_data(report_path)

    print(f"[OK] sole source: {report_path}")
    print(f"[OK] parsed cells: {len(data)}; status counts: {data['status'].value_counts().to_dict()}")
    for model, metadata in MODELS.items():
        stem = metadata["stem"]
        csv_path = output_root / "data" / f"{stem}.csv"
        pdf_path = output_root / "pdf" / f"{stem}.pdf"
        png_path = output_root / "png" / f"{stem}.png"
        caption_path = output_root / "captions" / f"{stem}.txt"
        model_data = write_model_csv(data, model, csv_path)
        render_model(model_data, model, pdf_path, png_path)
        write_caption(model, caption_path)
        print(f"[OK] {model}: {len(model_data)} rows")
        print(f"  CSV: {csv_path}")
        print(f"  PDF: {pdf_path}")
        print(f"  PNG: {png_path}")
        print(f"  caption: {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
