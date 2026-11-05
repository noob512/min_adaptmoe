#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot formal and end-to-end E5 ablations without pooling their protocols."""

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


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
FIGURES = SCRIPT.parents[1]
NAME = "fig_core_mechanism_ablation"
DEFAULT_REPORT = (
    REPO_ROOT
    / "evaluation/data/processed/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md"
)
DEFAULT_SUPPLEMENT = (
    REPO_ROOT
    / "experiments/eurosys_moe_offload_evaluation/artifacts/"
    "e5-e2e-incremental-ablation-once-20260916/summary.md"
)
MODEL = "Qwen3-30B-A3B FP8"

PATH_ORDER = ("H", "G", "D", "D_no_prebuild")
PATH_LABELS = {
    "H": "Eager remote",
    "G": "Decode Graph",
    "D": "Dual Graph + prebuild",
    "D_no_prebuild": "Dual Graph, no prebuild",
}
PATH_COLORS = {
    "H": "#A6A6A6",
    "G": "#56B4E9",
    "D": "#0072B2",
    "D_no_prebuild": "#E69F00",
}
PATH_HATCHES = {"H": "..", "G": "", "D": "//", "D_no_prebuild": "\\\\"}
EDGE = "#333333"

CSV_COLUMNS = (
    "protocol", "panel", "model", "batch_size", "item", "path_key",
    "metric", "value", "unit", "baseline_value", "optimized_value",
    "recomputed_value", "ratio_difference", "prefill_ms", "p_to_d_ms",
    "decode_ms", "decode_tokens_per_s", "e2e_ms", "sample_nature",
    "source_section", "validity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--csv", type=Path, default=FIGURES / "data" / f"{NAME}.csv")
    parser.add_argument("--pdf", type=Path, default=FIGURES / "pdf" / f"{NAME}.pdf")
    parser.add_argument("--png", type=Path, default=FIGURES / "png" / f"{NAME}.png")
    parser.add_argument(
        "--caption", type=Path, default=FIGURES / "captions" / f"{NAME}.txt"
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_number(value: str) -> float:
    parsed = float(value.replace(",", "").replace("×", "").strip())
    require(math.isfinite(parsed) and parsed >= 0, f"invalid number: {value}")
    return parsed


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def extract_table(text: str, header: list[str]) -> list[list[str]]:
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if split_row(line) == header), -1
    )
    require(start >= 0, f"table not found: {header}")
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = split_row(line)
        require(len(cells) == len(header), f"malformed row: {line}")
        rows.append(cells)
    require(bool(rows), f"empty table: {header}")
    return rows


def extract_e5(report: str) -> str:
    start = report.find("## 8. E5：四项核心机制消融")
    end = report.find("## 9. E6：Plan Class与切换摊销", start + 1)
    require(start >= 0 and end > start, "E5 section not found")
    return report[start:end]


def capture(section: str, pattern: str, label: str) -> tuple[float, ...]:
    match = re.search(pattern, section, flags=re.DOTALL)
    require(match is not None, f"values not found: {label}")
    return tuple(parse_number(value) for value in match.groups())


def parse_formal_ratios(section: str) -> dict[str, tuple[float, float]]:
    rows = extract_table(section, ["核心机制", "B1", "B4"])
    require(len(rows) == 5, "formal ratio row count differs")
    return {row[0]: (parse_number(row[1]), parse_number(row[2])) for row in rows}


def find_ratio(
    ratios: dict[str, tuple[float, float]], *fragments: str
) -> tuple[float, float]:
    matches = [
        values
        for label, values in ratios.items()
        if all(fragment in label for fragment in fragments)
    ]
    require(len(matches) == 1, f"ratio row not unique: {fragments}")
    return matches[0]


def empty_record() -> dict[str, object]:
    return {column: np.nan for column in CSV_COLUMNS}


def build_formal(report_path: Path) -> list[dict[str, object]]:
    section = extract_e5(report_path.read_text(encoding="utf-8"))
    require("模型为Qwen3-30B-A3B-FP8" in section, "formal model differs")
    require("每个性能臂3个独立冷进程" in section, "formal sample nature differs")
    require("不能把表中倍率相乘" in section, "dependency warning missing")
    ratios = parse_formal_ratios(section)

    resume_b1, resume_b4, rebuild_b1, rebuild_b4 = capture(
        section,
        r"双视图，RESUME为B1/B4\s+([0-9.,]+)/([0-9.,]+) ms；DESTROY/rebuild为\s+"
        r"([0-9.,]+)/([0-9.,]+) ms",
        "PDE switch",
    )
    runtime_b1, runtime_b4 = capture(
        section,
        r"runtime range-install RESUME为\s*([0-9.,]+)/([0-9.,]+) ms",
        "runtime install",
    )
    graph_values = capture(
        section,
        r"B1/B4的\s*Prefill从([0-9.,]+)/([0-9.,]+) ms降至([0-9.,]+)/([0-9.,]+) ms，"
        r"Decode从\s*([0-9.,]+)/([0-9.,]+) ms降至([0-9.,]+)/([0-9.,]+) ms",
        "Graph",
    )
    stable_b1, stable_b4, nonstable_b1, nonstable_b4 = capture(
        section,
        r"stable VA的B1/B4重建中位数为\s*([0-9.,]+)/([0-9.,]+) ms；"
        r"每代8/8地址变化并重捕获Graph时为\s*([0-9.,]+)/([0-9.,]+) ms",
        "formal stable VA",
    )
    (
        prefill_eager_b1, prefill_eager_b4, prefill_graph_b1, prefill_graph_b4,
        decode_eager_b1, decode_eager_b4, decode_graph_b1, decode_graph_b4,
    ) = graph_values

    items = (
        ("P/D direct switch", (resume_b1, resume_b4), (rebuild_b1, rebuild_b4),
         find_ratio(ratios, "PDE直接切换")),
        ("Prefill Graph", (prefill_graph_b1, prefill_graph_b4),
         (prefill_eager_b1, prefill_eager_b4), find_ratio(ratios, "CUDA Graph", "Prefill")),
        ("Decode Graph", (decode_graph_b1, decode_graph_b4),
         (decode_eager_b1, decode_eager_b4), find_ratio(ratios, "CUDA Graph", "Decode")),
        ("Stable Parameter VA", (stable_b1, stable_b4), (nonstable_b1, nonstable_b4),
         find_ratio(ratios, "地址稳定")),
        ("Prebuilt page tables", (resume_b1, resume_b4), (runtime_b1, runtime_b4),
         find_ratio(ratios, "页表提前构建")),
    )

    records: list[dict[str, object]] = []
    for item, optimized_values, baseline_values, reported_values in items:
        for index, batch_size in enumerate((1, 4)):
            optimized = optimized_values[index]
            baseline = baseline_values[index]
            recomputed = baseline / optimized
            validity = "valid_formal_three_cold_process_median"
            if item in {"P/D direct switch", "Prebuilt page tables"}:
                validity += "_dependent_not_multiplicative"
            if item in {"Prefill Graph", "Decode Graph"} and batch_size == 4:
                validity += "_cross_arm_hash_mismatch"
            record = empty_record()
            record.update(
                {
                    "protocol": "formal_three_cold_process", "panel": "a",
                    "model": MODEL, "batch_size": batch_size, "item": item,
                    "metric": "reported paired speedup", "value": reported_values[index],
                    "unit": "×", "baseline_value": baseline,
                    "optimized_value": optimized, "recomputed_value": recomputed,
                    "ratio_difference": abs(recomputed - reported_values[index]),
                    "sample_nature": "median of three independent cold processes",
                    "source_section": f"{report_path}: E5 formal four-mechanism ablation",
                    "validity": validity,
                }
            )
            records.append(record)
    return records


def build_supplement(summary_path: Path) -> list[dict[str, object]]:
    text = summary_path.read_text(encoding="utf-8")
    require("one cold process" in text, "supplement sample nature differs")
    path_rows = extract_table(
        text,
        ["B", "Path", "Prefill ms", "P→D ms", "Decode ms", "Decode tok/s", "Cycle E2E ms"],
    )
    require(len(path_rows) == 8, "incremental path row count differs")
    path_keys = {
        "eager/eager remote (H)": "H",
        "eager/Decode Graph remote (G)": "G",
        "Full Graph + prebuild (D)": "D",
        "Full Graph, no prebuild": "D_no_prebuild",
    }
    records: list[dict[str, object]] = []
    for row in path_rows:
        key = path_keys.get(row[1])
        require(key is not None, f"unknown path: {row[1]}")
        record = empty_record()
        record.update(
            {
                "protocol": "end_to_end_single_cold_process", "panel": "b,c",
                "model": MODEL, "batch_size": int(row[0]), "item": PATH_LABELS[key],
                "path_key": key, "metric": "incremental request-path metrics",
                "prefill_ms": parse_number(row[2]), "p_to_d_ms": parse_number(row[3]),
                "decode_ms": parse_number(row[4]),
                "decode_tokens_per_s": parse_number(row[5]), "e2e_ms": parse_number(row[6]),
                "sample_nature": "one independent cold process; descriptive point",
                "source_section": f"{summary_path}: end-to-end incremental ablation",
                "validity": (
                    "valid_descriptive_single_cold_process_"
                    "cross_arm_semantics_not_acceptance_gate"
                ),
            }
        )
        records.append(record)

    stable_rows = extract_table(
        text,
        ["B", "Stable rebuild ms", "Non-stable reallocate/recapture ms", "Non-stable/stable"],
    )
    require(len(stable_rows) == 2, "stable VA row count differs")
    for row in stable_rows:
        stable = parse_number(row[1])
        nonstable = parse_number(row[2])
        reported = parse_number(row[3])
        record = empty_record()
        record.update(
            {
                "protocol": "stable_va_single_cold_process", "panel": "d",
                "model": MODEL, "batch_size": int(row[0]), "item": "Stable VA recovery",
                "metric": "non-stable / stable recovery latency", "value": reported,
                "unit": "×", "baseline_value": nonstable, "optimized_value": stable,
                "recomputed_value": nonstable / stable,
                "ratio_difference": abs(nonstable / stable - reported),
                "sample_nature": "one independent cold process; descriptive point",
                "source_section": f"{summary_path}: Stable/non-stable VA rebuild",
                "validity": "valid_descriptive_single_cold_process",
            }
        )
        records.append(record)
    return records


def build_data(report_path: Path, summary_path: Path) -> pd.DataFrame:
    data = pd.DataFrame.from_records(
        build_formal(report_path) + build_supplement(summary_path), columns=CSV_COLUMNS
    )
    require(len(data) == 20, "combined ablation row count differs")
    require(
        set(data["protocol"]) == {
            "formal_three_cold_process", "end_to_end_single_cold_process",
            "stable_va_single_cold_process",
        },
        "protocol coverage differs",
    )
    return data


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4, "axes.titlesize": 8.4, "axes.labelsize": 8.2,
            "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
            "axes.linewidth": 0.8, "hatch.linewidth": 0.55,
            "pdf.fonttype": 42, "ps.fonttype": 42,
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#E1E1E1", linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", length=0, pad=3)


def formal_panel(axis: plt.Axes, data: pd.DataFrame) -> None:
    formal = data[data["protocol"] == "formal_three_cold_process"]
    items = (
        "P/D direct switch", "Prefill Graph", "Decode Graph",
        "Stable Parameter VA", "Prebuilt page tables",
    )
    display_labels = ("P/D switch", "Prefill Graph", "Decode Graph", "Stable VA", "Prebuilt views")
    centers = np.arange(len(items), dtype=float)
    width = 0.34
    values_by_batch: dict[int, np.ndarray] = {}
    styles = {
        1: {"color": "#0072B2", "hatch": "", "offset": -width / 2},
        4: {"color": "#009E73", "hatch": "//", "offset": width / 2},
    }
    for batch_size, bar_style in styles.items():
        values = np.array([
            float(formal[(formal["item"] == item) &
                         (formal["batch_size"] == batch_size)]["value"].iloc[0])
            for item in items
        ])
        values_by_batch[batch_size] = values
        axis.bar(
            centers + bar_style["offset"], values - 1.0, width=width,
            bottom=1.0, color=bar_style["color"], edgecolor=EDGE,
            linewidth=0.7, hatch=bar_style["hatch"], zorder=3,
        )
    for index, center in enumerate(centers):
        b1 = values_by_batch[1][index]
        b4 = values_by_batch[4][index]
        axis.text(
            center, max(b1, b4) * 1.28, f"{b1:.2f} / {b4:.2f}×",
            ha="center", va="bottom", fontsize=6.2,
        )
    axis.axhline(1, color="#6C7D88", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    axis.set_yscale("log", base=10)
    axis.set_ylim(0.95, 5000)
    axis.set_yticks((1, 10, 100, 1000), ("1×", "10×", "100×", "1000×"))
    axis.set_xticks(centers, ("P/D\nswitch", "Prefill\nGraph", "Decode\nGraph", "Stable\nVA", "Prebuilt\nviews"))
    axis.set_ylabel("Speedup over baseline")
    axis.set_title("(a) Formal paired speedups", loc="left", pad=4)
    axis.grid(axis="y", color="#E1E1E1", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", length=0, pad=3)
    axis.legend(
        handles=[
            Patch(facecolor="#0072B2", edgecolor=EDGE, linewidth=0.7, label="B=1"),
            Patch(facecolor="#009E73", edgecolor=EDGE, linewidth=0.7, hatch="//", label="B=4"),
        ],
        loc="upper right", ncol=2, frameon=False, handletextpad=0.4, columnspacing=0.8,
    )


def incremental_bars(
    axis: plt.Axes, data: pd.DataFrame, *, column: str, ylabel: str,
    title: str, ylim: tuple[float, float], callouts: tuple[str, str],
) -> None:
    incremental = data[data["protocol"] == "end_to_end_single_cold_process"]
    centers = np.arange(2, dtype=float)
    width = 0.18
    offsets = dict(zip(PATH_ORDER, (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)))
    for key in PATH_ORDER:
        values = [
            float(incremental[(incremental["batch_size"] == batch_size) &
                              (incremental["path_key"] == key)][column].iloc[0])
            for batch_size in (1, 4)
        ]
        axis.bar(
            centers + offsets[key], values, width=width * 0.92,
            color=PATH_COLORS[key], edgecolor=EDGE, linewidth=0.65,
            hatch=PATH_HATCHES[key], zorder=3,
        )
    axis.set_xticks(centers, ("B=1", "B=4"))
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    axis.set_title(title, loc="left", pad=4)
    for xpos, text in zip(centers, callouts, strict=True):
        axis.text(
            xpos, ylim[1] * 0.965, text, ha="center", va="top",
            fontsize=6.5, color="#444444",
        )
    style_axis(axis)


def stable_panel(axis: plt.Axes, data: pd.DataFrame) -> None:
    stable = data[data["protocol"] == "stable_va_single_cold_process"].sort_values("batch_size")
    centers = np.arange(2, dtype=float)
    width = 0.30
    stable_s = stable["optimized_value"].astype(float).to_numpy() / 1000.0
    nonstable_s = stable["baseline_value"].astype(float).to_numpy() / 1000.0
    axis.bar(
        centers - width / 2, stable_s, width, color="#0072B2", edgecolor=EDGE,
        linewidth=0.7, label="Stable VA", zorder=3,
    )
    axis.bar(
        centers + width / 2, nonstable_s, width, color="#E69F00", edgecolor=EDGE,
        linewidth=0.7, hatch="//", label="Reallocate + recapture", zorder=3,
    )
    for index, row in enumerate(stable.itertuples(index=False)):
        axis.text(
            centers[index], max(stable_s[index], nonstable_s[index]) + 0.30,
            f"{float(row.value):.2f}×", ha="center", va="bottom",
            fontsize=7.0, fontweight="bold",
        )
    axis.set_xticks(centers, ("B=1", "B=4"))
    axis.set_ylim(0, 9.0)
    axis.set_ylabel("Recovery latency (s)")
    axis.set_title("(d) Stable Parameter VA", loc="left", pad=4)
    axis.legend(loc="upper right", frameon=False, handlelength=1.3)
    style_axis(axis)


def render(data: pd.DataFrame, pdf: Path, png: Path) -> None:
    configure()
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.05))
    axis_a, axis_b, axis_c, axis_d = axes.flat
    formal_panel(axis_a, data)
    incremental_bars(
        axis_b, data, column="e2e_ms", ylabel="Cycle E2E latency (ms)",
        title="(b) Incremental end-to-end latency", ylim=(0, 2300),
        callouts=("H→G: 3.61×\nG→D: 1.52×", "H→G: 2.60×\nG→D: 1.27×"),
    )
    incremental_bars(
        axis_c, data, column="decode_tokens_per_s", ylabel="Decode throughput (tokens/s)",
        title="(c) Incremental Decode throughput", ylim=(0, 210),
        callouts=("H→G: 5.57×", "H→G: 2.90×"),
    )
    stable_panel(axis_d, data)
    figure.legend(
        handles=[
            Patch(
                facecolor=PATH_COLORS[key], edgecolor=EDGE, linewidth=0.65,
                hatch=PATH_HATCHES[key], label=PATH_LABELS[key],
            )
            for key in PATH_ORDER
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.002), ncol=4, frameon=False,
        columnspacing=1.0, handlelength=1.35,
    )
    figure.subplots_adjust(
        left=0.125, right=0.99, bottom=0.095, top=0.88, hspace=0.39, wspace=0.34
    )
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf, format="pdf")
    figure.savefig(png, format="png", dpi=300)
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "Ablation of AdaptMoE on Qwen3-30B-A3B FP8 at B=1 and B=4. "
        "(a) Formal paired ablations report medians from three independent cold "
        "processes; bars are the reported baseline/optimized latency ratios, and labels "
        "list B=1/B=4. Direct "
        "P/D switching depends on prebuilt page tables, so the two control-plane ratios "
        "are not additive or multiplicative. (b–c) A separate single-cold-process "
        "supplement incrementally enables Decode Graph (H→G), then Prefill/Decode Graphs "
        "with prebuilt fast P→D (G→D); G→D is a combined change, not a pure Prefill-Graph "
        "ablation. Disabling prebuild changes one-cycle E2E by only 1.15%/1.60% despite "
        "making RESUME 20.16×/17.23× slower. (d) Stable VAs reduce rebuild/recapture "
        "latency, although stable rebuild remains 233.16×/209.00× slower than direct P→D. "
        "Panels (b–d) are descriptive single-process points under a different protocol; "
        "they are not pooled with (a), and no error bars or confidence intervals are "
        "claimed. Cross-arm semantic equality was not an acceptance gate for the "
        "single-process supplement; the formal B=4 eager/Graph hashes also differ."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = build_data(args.report, args.supplement)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.csv, index=False, float_format="%.9f")
    render(data, args.pdf, args.png)
    write_caption(args.caption)
    formal = data[data["protocol"] == "formal_three_cold_process"]
    for item in formal["item"].drop_duplicates():
        values = formal[formal["item"] == item].sort_values("batch_size")["value"]
        print(f"{item}: B1={values.iloc[0]:.2f}x, B4={values.iloc[1]:.2f}x")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png} (300 dpi)")
    print(f"Wrote {args.caption}")


if __name__ == "__main__":
    main()
