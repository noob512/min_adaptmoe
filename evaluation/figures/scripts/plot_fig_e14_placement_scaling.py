#!/usr/bin/env python3
"""Build Figure E14 from the authoritative structured formal artifact.

The script never contains experimental measurements. It extracts one tidy row
per independent cold-process median from result.json, validates the extracted
cross-process medians against verification.json and summary.csv, writes the
tidy CSV, and then renders the PDF and PNG from that CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
PAPER_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "experiments/eurosys_moe_offload_evaluation/artifacts/"
    "e14-uva-eager-request-cold29-direct-formal-20260903-attempt-01"
)
DEFAULT_TIDY = PAPER_ROOT / "figures/data/fig_e14_placement_scaling.csv"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "figures/output"

BATCHES = (1, 4, 16, 32)
PHASES = ("prefill", "decode")
EXPECTED_PROCESSES_PER_SCHEME = 3

# These styles retain the paper-wide deep-blue/filled-circle and
# orange/open-circle identities.
SCHEMES = {
    "e14_pure_uva_distributed_cold29": {
        "name": "Layer-wise cold-expert offload",
        "short": "Layer-wise",
        "color": "#0072B2",
        "marker": "o",
        "marker_face": "#0072B2",
    },
    "e14_pure_uva_fixed_whole_layers_0_10": {
        "name": "Whole-layer offload",
        "short": "Whole-layer",
        "color": "#E69F00",
        "marker": "o",
        "marker_face": "white",
    },
}
SCHEME_ORDER = tuple(item["name"] for item in SCHEMES.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE,
        help="authoritative E14 artifact directory containing result.json files",
    )
    parser.add_argument(
        "--tidy-csv",
        type=Path,
        default=DEFAULT_TIDY,
        help="derived tidy CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for PDF and PNG",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float, *, tolerance: float = 5e-7) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def extract_tidy_rows(source_root: Path) -> list[dict[str, Any]]:
    verification_path = source_root / "verification.json"
    verification = load_json(verification_path)
    require(verification.get("status") == "ok", "verification.json is not status=ok")
    protocol = verification.get("protocol") or {}
    require(
        tuple(int(value) for value in protocol.get("batch_sizes", [])) == BATCHES,
        "verification batch matrix differs from B1/B4/B16/B32",
    )
    require(
        protocol.get("formal_process_count_met") is True,
        "formal independent-process count gate did not pass",
    )
    require(
        (verification.get("cleanup") or {}).get("stock_uvm_restored") is True
        and (verification.get("cleanup") or {}).get("final_gpu_clean") is True,
        "artifact cleanup gate did not pass",
    )

    result_paths = sorted(source_root.glob("*/result.json"))
    require(
        len(result_paths) == 2 * EXPECTED_PROCESSES_PER_SCHEME,
        f"expected 6 process result files, found {len(result_paths)}",
    )

    paths_by_scheme: dict[str, list[Path]] = defaultdict(list)
    payload_by_path: dict[Path, dict[str, Any]] = {}
    for result_path in result_paths:
        result = load_json(result_path)
        require(result.get("status") == "ok", f"failed process artifact: {result_path}")
        mechanism = str(result.get("mechanism_class"))
        require(mechanism in SCHEMES, f"unexpected E14 mechanism: {mechanism}")
        require(
            tuple(
                sorted(
                    int(value)
                    for value in (result.get("config") or {}).get("batch_sizes", [])
                )
            )
            == BATCHES,
            f"batch matrix differs in {result_path}",
        )
        require(
            int((result.get("config") or {}).get("measurement_repeats", 0)) == 3,
            f"measurement repeat count differs in {result_path}",
        )
        paths_by_scheme[mechanism].append(result_path)
        payload_by_path[result_path] = result

    rows: list[dict[str, Any]] = []
    for mechanism, style in SCHEMES.items():
        paths = paths_by_scheme[mechanism]
        require(
            len(paths) == EXPECTED_PROCESSES_PER_SCHEME,
            f"{style['name']}: expected 3 independent processes, found {len(paths)}",
        )
        for replicate, result_path in enumerate(paths, start=1):
            result = payload_by_path[result_path]
            summaries = result.get("phase_summary") or {}
            for batch in BATCHES:
                summary = summaries.get(str(batch))
                require(isinstance(summary, dict), f"missing B{batch}: {result_path}")
                require(
                    int(summary.get("measurement_repeats", 0)) == 3,
                    f"B{batch} is not a three-repeat process median: {result_path}",
                )
                prefill_ms = float(summary["prefill_step_ms"]["median"])
                decode_ms = float(summary["decode_repeat_median_ms"]["median"])
                prompt_tokens = int(summary["prompt_tokens"])
                require(prefill_ms > 0 and decode_ms > 0, f"non-positive latency: {result_path}")
                measurements = {
                    "prefill": prompt_tokens * 1000.0 / prefill_ms,
                    "decode": batch * 1000.0 / decode_ms,
                }
                for phase, throughput in measurements.items():
                    require(
                        math.isfinite(throughput) and throughput > 0,
                        f"invalid {phase} throughput: {result_path}",
                    )
                    rows.append(
                        {
                            "experiment": "E14",
                            "phase": phase,
                            "batch_size": batch,
                            "scheme": style["name"],
                            "replicate": replicate,
                            "throughput_tokens_per_second": f"{throughput:.9f}",
                            "within_process_measurements": int(summary["measurement_repeats"]),
                            "within_process_statistic": "median",
                            "source_result_json": str(result_path.relative_to(REPO_ROOT)),
                            "source_sha256": sha256(result_path),
                        }
                    )
    return rows


def write_tidy_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_tidy_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 48, f"tidy CSV must contain 48 observations, found {len(rows)}")
    return rows


def aggregate_tidy(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, float]]:
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["scheme"], row["phase"], int(row["batch_size"]))
        values[key].append(float(row["throughput_tokens_per_second"]))

    aggregate: dict[tuple[str, str, int], dict[str, float]] = {}
    for scheme in SCHEME_ORDER:
        for phase in PHASES:
            for batch in BATCHES:
                samples = values[(scheme, phase, batch)]
                require(
                    len(samples) == EXPECTED_PROCESSES_PER_SCHEME,
                    f"{scheme}/{phase}/B{batch}: expected three process medians",
                )
                aggregate[(scheme, phase, batch)] = {
                    "median": statistics.median(samples),
                    "min": min(samples),
                    "max": max(samples),
                }
    return aggregate


def validate_against_authoritative_aggregate(
    source_root: Path,
    aggregate: dict[tuple[str, str, int], dict[str, float]],
) -> None:
    verification = load_json(source_root / "verification.json")
    arm_for_scheme = {SCHEME_ORDER[0]: "D", SCHEME_ORDER[1]: "W"}
    metric_for_phase = {
        "prefill": "prefill_tokens_per_second",
        "decode": "decode_tokens_per_second",
    }
    official = verification["aggregate_process_medians"]
    official_scaling = verification["scaling"]

    for scheme in SCHEME_ORDER:
        arm = arm_for_scheme[scheme]
        for phase in PHASES:
            metric = metric_for_phase[phase]
            b1 = aggregate[(scheme, phase, 1)]["median"]
            for batch in BATCHES:
                observed = aggregate[(scheme, phase, batch)]["median"]
                expected = float(official[arm][str(batch)][metric]["median"])
                require(
                    close(observed, expected),
                    f"aggregate mismatch for {scheme}/{phase}/B{batch}: "
                    f"{observed} != {expected}",
                )
                efficiency = observed / (batch * b1)
                expected_efficiency = float(
                    official_scaling[arm][str(batch)][f"{phase}_scaling_efficiency"]
                )
                require(
                    close(efficiency, expected_efficiency),
                    f"scaling-efficiency mismatch for {scheme}/{phase}/B{batch}",
                )

    summary_rows: dict[tuple[str, int], dict[str, str]] = {}
    with (source_root / "summary.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            summary_rows[(row["arm"], int(row["batch"]))] = row
    arm_name = {
        SCHEME_ORDER[0]: "distributed_cold29",
        SCHEME_ORDER[1]: "fixed_whole_layers_0_10",
    }
    for scheme in SCHEME_ORDER:
        for batch in BATCHES:
            source_row = summary_rows[(arm_name[scheme], batch)]
            for phase, column in (("prefill", "prefill_tok_s"), ("decode", "decode_tok_s")):
                observed = aggregate[(scheme, phase, batch)]["median"]
                require(
                    close(observed, float(source_row[column])),
                    f"summary.csv mismatch for {scheme}/{phase}/B{batch}",
                )


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.0,
            "lines.linewidth": 1.2,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def scheme_style(scheme: str) -> dict[str, str]:
    for style in SCHEMES.values():
        if style["name"] == scheme:
            return style
    raise KeyError(scheme)


def render_figure(
    aggregate: dict[tuple[str, str, int], dict[str, float]], output_dir: Path
) -> tuple[Path, Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    phase_titles = {"prefill": "(a) Prefill", "decode": "(b) Decode"}

    for ax, phase in zip(axes, PHASES, strict=True):
        plotted_max = 0.0
        for scheme in SCHEME_ORDER:
            style = scheme_style(scheme)
            medians = [aggregate[(scheme, phase, batch)]["median"] for batch in BATCHES]
            lowers = [
                aggregate[(scheme, phase, batch)]["median"]
                - aggregate[(scheme, phase, batch)]["min"]
                for batch in BATCHES
            ]
            uppers = [
                aggregate[(scheme, phase, batch)]["max"]
                - aggregate[(scheme, phase, batch)]["median"]
                for batch in BATCHES
            ]
            # At paper scale the Prefill/B16 range stem is visually confused
            # with an extra mark below the filled data point.  Show only the
            # measured median marker at that coordinate, as requested.
            if phase == "prefill" and scheme == "Layer-wise cold-expert offload":
                batch_16_index = BATCHES.index(16)
                lowers[batch_16_index] = 0.0
                uppers[batch_16_index] = 0.0
            plotted_max = max(
                plotted_max,
                max(median + upper for median, upper in zip(medians, uppers)),
            )
            ax.errorbar(
                BATCHES,
                medians,
                yerr=[lowers, uppers],
                color=style["color"],
                linestyle="-",
                linewidth=1.25,
                marker=style["marker"],
                markersize=5.2,
                markerfacecolor=style["marker_face"],
                markeredgecolor=style["color"],
                markeredgewidth=1.05,
                elinewidth=1.0,
                # Keep the measured min--max stem, but omit horizontal caps;
                # the lower cap at Prefill/B16 otherwise looks like a stray
                # line detached from the data marker at paper scale.
                capsize=0.0,
                zorder=3,
            )

        ax.set_xscale("log", base=2)
        ax.set_xlim(0.78, 41.0)
        ax.set_xticks(BATCHES, labels=[str(batch) for batch in BATCHES])
        ax.set_xlabel("Batch size (requests)")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title(phase_titles[phase], pad=3.5)
        ax.set_ylim(0.0, plotted_max * 1.12)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = []
    for scheme in SCHEME_ORDER:
        style = scheme_style(scheme)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=style["marker_face"],
                markeredgecolor=style["color"],
                markeredgewidth=1.05,
                linewidth=1.25,
                markersize=5.2,
                label=scheme,
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.087, right=0.992, bottom=0.205, top=0.795, wspace=0.29)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_e14_placement_scaling.pdf"
    png_path = output_dir / "fig_e14_placement_scaling.png"
    metadata = {
        "Title": "E14 expert-offload placement scaling",
        "Subject": "Absolute Prefill and Decode throughput versus batch size",
        "Creator": "plot_fig_e14_placement_scaling.py",
    }
    fig.savefig(pdf_path, format="pdf", metadata=metadata)
    fig.savefig(png_path, format="png", dpi=300, metadata={"Software": metadata["Creator"]})
    plt.close(fig)
    return pdf_path, png_path


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    tidy_path = args.tidy_csv.resolve()
    output_dir = args.output_dir.resolve()
    require(source_root.is_dir(), f"source artifact does not exist: {source_root}")

    rows = extract_tidy_rows(source_root)
    write_tidy_csv(tidy_path, rows)
    # Render only from the serialized tidy data, not directly from result.json.
    tidy_rows = read_tidy_csv(tidy_path)
    aggregate = aggregate_tidy(tidy_rows)
    validate_against_authoritative_aggregate(source_root, aggregate)
    pdf_path, png_path = render_figure(aggregate, output_dir)

    print(f"[OK] source: {source_root}")
    print(f"[OK] tidy CSV: {tidy_path} ({len(tidy_rows)} rows)")
    print("[OK] all 16 plotted medians match the structured sources")
    print(f"[OK] PDF: {pdf_path}")
    print(f"[OK] PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
