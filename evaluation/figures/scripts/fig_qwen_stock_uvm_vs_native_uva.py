#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot Qwen latency for the 6-GiB-KV Native UVA versus Stock UVM run."""

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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


SCRIPT_PATH = Path(__file__).resolve()
FIGURES_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURE_NAME = "fig_qwen_stock_uvm_vs_native_uva"

DEFAULT_SUMMARY = (
    REPO_ROOT
    / "evaluation/data/runs/Qwen_kv6_uva_vs_stock_uvm_20260912_single_round/summary.json"
)
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

CONFIGS = (
    ("short_dialogue", "Short", 1),
    ("short_dialogue", "Short", 16),
    ("ordinary_chat", "Ordinary", 1),
    ("ordinary_chat", "Ordinary", 16),
    ("medium_context", "Medium", 1),
    ("medium_context", "Medium", 16),
    ("rag_summary", "RAG", 1),
    ("rag_summary", "RAG", 16),
    ("long_generation", "Long", 1),
    ("long_generation", "Long", 16),
)
PLOT_CONFIGS = tuple(config for config in CONFIGS if config[2] == 1)
METRICS = (
    ("ttft", "TTFT", "(a) TTFT"),
    ("tpot", "TPOT", "(b) TPOT"),
    ("e2e", "E2E", "(c) End-to-end latency"),
)
MODES = ("native_uva_6g", "stock_uvm_demand")
EXPECTED_UVA_OOM = {("medium_context", 16), ("rag_summary", 16)}
EXPECTED_SCHEMA = "qwen_kv6_uva_vs_stock_uvm_summary_v1"
KV_CACHE_BYTES = 6 * 1024**3

NATIVE_COLOR = "#0072B2"
STOCK_COLOR = "#E69F00"
BAR_WIDTH = 0.35
BAR_OFFSET = 0.19

CSV_COLUMNS = [
    "workload",
    "concurrency",
    "native_uva_status",
    "stock_uvm_status",
    "native_uva_ttft_ms",
    "stock_uvm_ttft_ms",
    "ttft_ratio",
    "native_uva_tpot_ms",
    "stock_uvm_tpot_ms",
    "tpot_ratio",
    "native_uva_e2e_ms",
    "stock_uvm_e2e_ms",
    "e2e_ratio",
    "kv_cache_gib",
    "native_uva_actual_offload_gib",
    "outputs_match",
    "source_section",
    "validity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--caption", type=Path, default=DEFAULT_CAPTION)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive_number(value: object, label: str) -> float:
    require(isinstance(value, (int, float)), f"{label} is not numeric")
    number = float(value)
    require(math.isfinite(number) and number > 0, f"{label} is not positive and finite")
    return number


def validate_summary(summary: dict[str, object]) -> None:
    require(summary.get("schema") == EXPECTED_SCHEMA, "unexpected summary schema")
    require(summary.get("kv_cache_bytes") == KV_CACHE_BYTES, "KV cache is not exactly 6 GiB")
    require(summary.get("kv_cache_gib") == 6, "KV cache GiB field differs")
    require(summary.get("cells") == 20, "expected 20 experiment cells")
    require(summary.get("logical_pairs") == 10, "expected 10 logical pairs")
    require(summary.get("uva_ok") == 8, "expected 8 successful Native UVA cells")
    require(summary.get("uvm_ok") == 10, "expected 10 successful Stock UVM cells")
    require(summary.get("comparable_pairs") == 8, "expected 8 comparable pairs")
    require(summary.get("matching_output_pairs") == 8, "expected 8 matching-output pairs")
    require(summary.get("complete_protocol") is True, "protocol is incomplete")
    audit = summary.get("uvm_allocator_audit")
    require(isinstance(audit, dict), "UVM allocator audit is missing")
    require(audit.get("log_count") == 10, "expected 10 UVM allocator logs")
    require(audit.get("all_required_disable_flags_present") is True, "UVM disable flags differ")
    require(audit.get("explicit_prefetch_or_advice_matches") == 0, "prefetch/advice was observed")
    require(audit.get("all_have_managed_allocation_trace") is True, "managed traces are incomplete")


def build_data(summary_path: Path) -> pd.DataFrame:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(isinstance(summary, dict), "summary root is not an object")
    validate_summary(summary)

    rows = summary.get("rows")
    pairs = summary.get("pairs")
    require(isinstance(rows, list) and len(rows) == 20, "summary rows differ")
    require(isinstance(pairs, list) and len(pairs) == 10, "summary pairs differ")

    raw_rows: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in rows:
        require(isinstance(row, dict), "a summary row is not an object")
        key = (str(row.get("workload")), int(row.get("concurrency")), str(row.get("mode")))
        require(key not in raw_rows, f"duplicate summary row: {key}")
        require(row.get("kv_cache_bytes") == KV_CACHE_BYTES, f"KV cache differs for {key}")
        raw_rows[key] = row

    expected_keys = {
        (workload_key, concurrency, mode)
        for workload_key, _workload, concurrency in CONFIGS
        for mode in MODES
    }
    require(set(raw_rows) == expected_keys, "summary row coverage differs")

    pair_rows: dict[tuple[str, int], dict[str, object]] = {}
    for pair in pairs:
        require(isinstance(pair, dict), "a pair row is not an object")
        key = (str(pair.get("workload")), int(pair.get("concurrency")))
        require(key not in pair_rows, f"duplicate pair row: {key}")
        pair_rows[key] = pair
    require(
        set(pair_rows) == {(key, concurrency) for key, _name, concurrency in CONFIGS},
        "pair coverage differs",
    )

    observed_source_oom = {
        (workload_key, concurrency)
        for workload_key, _workload, concurrency in CONFIGS
        if raw_rows[(workload_key, concurrency, "native_uva_6g")].get("status") == "OOM"
    }
    require(observed_source_oom == EXPECTED_UVA_OOM, "Native UVA source OOM coverage differs")

    records: list[dict[str, object]] = []
    for workload_key, workload, concurrency in PLOT_CONFIGS:
        key = (workload_key, concurrency)
        native = raw_rows[(workload_key, concurrency, "native_uva_6g")]
        stock = raw_rows[(workload_key, concurrency, "stock_uvm_demand")]
        pair = pair_rows[key]
        native_status = str(native.get("status"))
        stock_status = str(stock.get("status"))
        require(stock_status == "ok", f"Stock UVM is not valid for {key}")
        require(stock.get("returncode") == 0, f"Stock UVM return code differs for {key}")
        if native_status == "OOM":
            require(native.get("returncode") != 0, f"Native UVA OOM returned zero for {key}")
            require(pair.get("comparable") is False, f"OOM pair marked comparable for {key}")
            require(pair.get("outputs_match") is None, f"OOM pair has output match for {key}")
        else:
            require(native_status == "ok", f"unexpected Native UVA status for {key}")
            require(native.get("returncode") == 0, f"Native UVA return code differs for {key}")
            require(pair.get("comparable") is True, f"valid pair is not comparable for {key}")
            require(pair.get("outputs_match") is True, f"outputs do not match for {key}")

        record: dict[str, object] = {
            "workload": workload,
            "concurrency": concurrency,
            "native_uva_status": native_status,
            "stock_uvm_status": stock_status,
            "kv_cache_gib": 6.0,
            "native_uva_actual_offload_gib": positive_number(
                native.get("actual_offload_gib"), f"Native UVA actual offload/{key}"
            ),
            "outputs_match": pair.get("outputs_match"),
            "source_section": (
                "Qwen_kv6_uva_vs_stock_uvm_20260912_single_round/summary.json; "
                f"schema={EXPECTED_SCHEMA}"
            ),
            "validity": (
                "native_uva_oom_stock_uvm_valid_single_run"
                if native_status == "OOM"
                else "valid_matching_outputs_single_run_unpaired"
            ),
        }
        for prefix, _source_metric, _panel_title in METRICS:
            stock_ms = positive_number(stock.get(f"{prefix}_ms"), f"Stock UVM/{key}/{prefix}")
            native_value = native.get(f"{prefix}_ms")
            if native_status == "OOM":
                require(native_value is None, f"OOM Native UVA has {prefix} value for {key}")
                native_ms = math.nan
                ratio = math.nan
                require(pair.get(f"uvm_over_uva_{prefix}") is None, f"OOM pair has ratio for {key}")
            else:
                native_ms = positive_number(native_value, f"Native UVA/{key}/{prefix}")
                ratio = stock_ms / native_ms
                reported_ratio = positive_number(
                    pair.get(f"uvm_over_uva_{prefix}"), f"reported ratio/{key}/{prefix}"
                )
                require(
                    math.isclose(ratio, reported_ratio, rel_tol=1e-12, abs_tol=0),
                    f"ratio differs from source milliseconds for {key}/{prefix}",
                )
            record[f"native_uva_{prefix}_ms"] = native_ms
            record[f"stock_uvm_{prefix}_ms"] = stock_ms
            record[f"{prefix}_ratio"] = ratio
        records.append(record)

    data = pd.DataFrame.from_records(records)
    require(len(data) == 5, "expected five C1 plot rows")
    return data[CSV_COLUMNS]


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.8f", na_rep="")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == CSV_COLUMNS, "serialized CSV columns differ")
    require(len(serialized) == 5, "serialized CSV row count differs")
    oom = serialized["native_uva_status"] == "OOM"
    require(oom.sum() == 0, "C1 CSV unexpectedly contains OOM")
    for prefix, _source_metric, _panel_title in METRICS:
        require(serialized.loc[oom, f"native_uva_{prefix}_ms"].isna().all(), "OOM became numeric")
        require(serialized.loc[oom, f"{prefix}_ratio"].isna().all(), "OOM ratio became numeric")
        valid = ~oom
        expected = (
            serialized.loc[valid, f"stock_uvm_{prefix}_ms"]
            / serialized.loc[valid, f"native_uva_{prefix}_ms"]
        )
        require(
            np.allclose(serialized.loc[valid, f"{prefix}_ratio"], expected, rtol=0, atol=5e-9),
            f"serialized {prefix} ratios differ from source milliseconds",
        )
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.3,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.5,
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
    group_x = np.arange(len(PLOT_CONFIGS), dtype=float)
    native_x = group_x - BAR_OFFSET
    stock_x = group_x + BAR_OFFSET
    labels = [f"{workload}\nC{concurrency}" for _key, workload, concurrency in PLOT_CONFIGS]
    oom_mask = data["native_uva_status"].eq("OOM").to_numpy()

    figure, axes = plt.subplots(3, 1, figsize=(7.0, 5.3), sharex=True)
    for axis, (prefix, _source_metric, panel_title) in zip(axes, METRICS, strict=True):
        native = data[f"native_uva_{prefix}_ms"].to_numpy(dtype=float)
        stock = data[f"stock_uvm_{prefix}_ms"].to_numpy(dtype=float)
        ratios = data[f"{prefix}_ratio"].to_numpy(dtype=float)
        valid_native = np.isfinite(native)
        all_values = np.concatenate((native[valid_native], stock))
        lower = float(all_values.min()) / 2.5
        upper = float(all_values.max()) * 2.5
        baseline = 1.0

        axis.bar(
            native_x[valid_native],
            native[valid_native] - baseline,
            width=BAR_WIDTH,
            bottom=baseline,
            color=NATIVE_COLOR,
            edgecolor="none",
            zorder=3,
        )
        axis.bar(
            stock_x,
            stock - baseline,
            width=BAR_WIDTH,
            bottom=baseline,
            color=STOCK_COLOR,
            edgecolor="none",
            zorder=3,
        )
        for index, ratio in enumerate(ratios):
            if not math.isfinite(ratio):
                continue
            if stock[index] >= native[index]:
                label_x, label_y = stock_x[index], stock[index]
            else:
                label_x, label_y = native_x[index], native[index]
            axis.annotate(
                f"{ratio:.2f}×",
                xy=(label_x, label_y),
                xytext=(0, 2.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#333333",
                clip_on=False,
                zorder=4,
            )
        for index in np.flatnonzero(oom_mask):
            axis.text(
                native_x[index],
                lower * 1.18,
                "OOM",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color=NATIVE_COLOR,
                clip_on=True,
                zorder=4,
            )

        for separator in np.arange(len(PLOT_CONFIGS) - 1, dtype=float) + 0.5:
            axis.axvline(separator, color="#D8D8D8", linewidth=0.7, zorder=1)
        axis.set_yscale("log")
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        axis.yaxis.set_major_formatter(FuncFormatter(latency_tick))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_ylabel("Latency (ms)")
        axis.set_title(panel_title, loc="left", pad=3.0)
        axis.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.55, alpha=0.8)
        axis.grid(axis="y", which="minor", visible=False)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[-1].set_xticks(group_x, labels=labels)
    axes[-1].set_xlabel("Workload / Concurrency")
    axes[0].tick_params(axis="x", which="both", labelbottom=False)
    axes[1].tick_params(axis="x", which="both", labelbottom=False)

    legend_handles = [
        Line2D([0], [0], color=NATIVE_COLOR, linewidth=6.0, label="Native UVA"),
        Line2D(
            [0],
            [0],
            color=STOCK_COLOR,
            linewidth=6.0,
            label="Stock UVM demand paging",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.5,
    )
    figure.subplots_adjust(left=0.105, right=0.992, bottom=0.13, top=0.925, hspace=0.42)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Qwen 6-GiB-KV Stock UVM versus Native UVA latency",
        "Subject": "Absolute TTFT, TPOT, and end-to-end latency in milliseconds",
        "Creator": SCRIPT_PATH.name,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(
        png_path,
        format="png",
        dpi=300,
        metadata={"Software": SCRIPT_PATH.name},
    )
    plt.close(figure)


def metric_range(data: pd.DataFrame, prefix: str) -> tuple[float, float]:
    values = data[f"{prefix}_ratio"].dropna()
    return float(values.min()), float(values.max())


def write_caption(data: pd.DataFrame, path: Path) -> None:
    ttft_min, ttft_max = metric_range(data, "ttft")
    tpot_min, tpot_max = metric_range(data, "tpot")
    e2e_min, e2e_max = metric_range(data, "e2e")
    caption = (
        "Latency comparison between Stock UVM demand paging and Native UVA on "
        "Qwen3-30B-A3B FP8 with a 6 GiB KV cache. Bars report request-level P50 TTFT "
        "and TPOT plus batch completion latency (the last request's E2E), all in "
        "milliseconds; labels report Stock UVM / Native UVA for the five valid, "
        "output-matching C1 pairs. Native UVA offloads 6.20 GiB at whole-parameter "
        "granularity. Across the Short, Ordinary, Medium, RAG, and Long C1 workloads, "
        f"ratios span {ttft_min:.2f}–{ttft_max:.2f}× "
        f"for TTFT, {tpot_min:.2f}–{tpot_max:.2f}× for TPOT, and "
        f"{e2e_min:.2f}–{e2e_max:.2f}× for E2E. Each cell is one cold-process run with "
        "three warmup requests; no independent repeats were collected, so no error bars "
        "or confidence intervals are shown."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_statistics(data: pd.DataFrame) -> None:
    for prefix, _source_metric, _panel_title in METRICS:
        minimum, maximum = metric_range(data, prefix)
        print(f"{prefix.upper()} ratio min/max (5 C1 pairs): {minimum:.4f}× / {maximum:.4f}×")
    print("Plotted concurrency: C1 only")


def main() -> int:
    args = parse_args()
    summary_path = args.summary.resolve()
    csv_path = args.csv.resolve()
    pdf_path = args.pdf.resolve()
    png_path = args.png.resolve()
    caption_path = args.caption.resolve()
    require(summary_path.is_file(), f"summary not found: {summary_path}")

    data = build_data(summary_path)
    plot_data = write_and_reload_csv(data, csv_path)
    render_figure(plot_data, pdf_path, png_path)
    write_caption(plot_data, caption_path)

    print(f"[OK] source: {summary_path}")
    print(f"[OK] CSV: {csv_path} ({len(plot_data)} Qwen-only rows)")
    print(f"[OK] PDF: {pdf_path}")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] caption: {caption_path}")
    print_statistics(plot_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
