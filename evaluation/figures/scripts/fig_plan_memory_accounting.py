#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "pandas>=2.0,<4",
# ]
# ///
"""Plot audited expert/KV physical-page accounting for an E13 plan round trip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
FIGURES_ROOT = SCRIPT_PATH.parents[1]
FIGURE_NAME = "fig_plan_memory_accounting"

DEFAULT_RESULT = (
    REPO_ROOT
    / "experiments/eurosys_moe_offload_evaluation/artifacts/"
    "e13-plan-resize-physical-accounting-20260913-attempt-03/roundtrip/result.json"
)
DEFAULT_CSV = FIGURES_ROOT / "data" / f"{FIGURE_NAME}.csv"
DEFAULT_PDF = FIGURES_ROOT / "pdf" / f"{FIGURE_NAME}.pdf"
DEFAULT_PNG = FIGURES_ROOT / "png" / f"{FIGURE_NAME}.png"
DEFAULT_CAPTION = FIGURES_ROOT / "captions" / f"{FIGURE_NAME}.txt"

PLAN_NAMES = {"cold70": "K2", "cold80": "K4", "cold89": "K6"}
TRANSITIONS = ("K2→K4", "K4→K6", "K6→K4", "K4→K2")
COMPONENTS = ("Expert GPU pages", "KV GPU pages", "Net accounted pages")
COLORS = {
    "Expert GPU pages": "#0072B2",
    "KV GPU pages": "#E69F00",
    "Net accounted pages": "#666666",
}
HATCHES = {
    "Expert GPU pages": "",
    "KV GPU pages": "//",
    "Net accounted pages": "xx",
}
EDGE_COLOR = "#333333"
ACCOUNTING_PAGE_BYTES = 64 * 1024
BYTES_PER_GIB = 2**30
CSV_COLUMNS = (
    "scope",
    "order",
    "transition",
    "from_plan",
    "to_plan",
    "component",
    "change_bytes",
    "accounting_page_bytes",
    "change_pages",
    "change_gib",
    "sample_nature",
    "source_section",
    "validity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--caption", type=Path, default=DEFAULT_CAPTION)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_result(result_path: Path) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    require(payload.get("schema") == "eurosys_e13_k2_k6_roundtrip_v1",
            "unexpected result schema")
    require(payload.get("status") == "ok", "round-trip result is not valid")
    require(payload.get("arm") == "roundtrip", "unexpected result arm")
    require(isinstance(payload.get("transitions"), list), "transitions are missing")
    return payload


def validate_kv_snapshot(snapshot: dict[str, Any], expected_va_hash: str | None) -> str:
    require(snapshot.get("schema") == "vllm_uvm_kv_physical_snapshot_v1",
            "unexpected KV physical-snapshot schema")
    full = snapshot["full"]
    head = snapshot["head"]
    tail = snapshot["tail"]
    require(bool(full["ok"]) and bool(head["ok"]) and bool(tail["ok"]),
            "invalid KV physical snapshot")
    require(int(full["unknown_bytes"]) == 0, "KV snapshot contains unknown bytes")
    require(int(head["cpu_bytes"]) == 0, "KV active prefix is not GPU-only")
    require(int(tail["gpu_bytes"]) == 0, "KV inactive tail is not CPU-only")
    require(int(full["any_bytes"]) == int(snapshot["covered_bytes"]),
            "KV physical snapshot does not cover the full reservation")
    va_hash = str(snapshot["va_sha256"])
    if expected_va_hash is not None:
        require(va_hash == expected_va_hash, "KV fixed-VA hash changed")
    return va_hash


def add_component_records(
    records: list[dict[str, object]],
    *,
    scope: str,
    order: int,
    transition: str,
    from_plan: str,
    to_plan: str,
    changes: dict[str, int],
    source_sections: dict[str, str],
) -> None:
    for component in COMPONENTS:
        change_bytes = int(changes[component])
        require(change_bytes % ACCOUNTING_PAGE_BYTES == 0,
                f"{transition} {component} is not divisible by 64 KiB")
        records.append(
            {
                "scope": scope,
                "order": order,
                "transition": transition,
                "from_plan": from_plan,
                "to_plan": to_plan,
                "component": component,
                "change_bytes": change_bytes,
                "accounting_page_bytes": ACCOUNTING_PAGE_BYTES,
                "change_pages": change_bytes // ACCOUNTING_PAGE_BYTES,
                "change_gib": change_bytes / BYTES_PER_GIB,
                "sample_nature": "single audited qualification round trip",
                "source_section": source_sections[component],
                "validity": (
                    "valid_derived_cumulative_from_structured_audit"
                    if scope == "cumulative"
                    else (
                        "valid_structured_before_after_audit"
                        if component != "Net accounted pages"
                        else "valid_derived_expert_plus_kv_only"
                    )
                ),
            }
        )


def build_data(result_path: Path) -> pd.DataFrame:
    payload = load_result(result_path)
    raw_transitions = payload["transitions"]
    require(len(raw_transitions) == 5, "expected one no-op and four Plan transitions")

    records: list[dict[str, object]] = []
    expected_va_hash: str | None = None
    transition_changes: dict[str, dict[str, int]] = {}

    for raw in raw_transitions:
        before_plan_class = str(raw["before_plan"]["active_plan_class"])
        after_plan_class = str(raw["after_plan"]["active_plan_class"])
        require(before_plan_class in PLAN_NAMES and after_plan_class in PLAN_NAMES,
                "unknown Plan Class")

        for side in ("before_kv_physical", "after_kv_physical"):
            expected_va_hash = validate_kv_snapshot(raw[side], expected_va_hash)

        if before_plan_class == after_plan_class:
            require(raw["result"]["reason"] == "already_at_target", "unexpected no-op")
            continue

        transition = f"{PLAN_NAMES[before_plan_class]}→{PLAN_NAMES[after_plan_class]}"
        require(transition in TRANSITIONS, f"unexpected transition: {transition}")
        require(bool(raw["result"]["ok"]) and raw["result"]["reason"] == "applied",
                f"transition failed: {transition}")

        # More selected bytes mean that more expert bytes are cold/offloaded. Therefore,
        # the negative selected-byte change is the GPU-resident expert-byte change.
        expert_change = int(raw["before_plan"]["selected_parameter_bytes_per_forward"]) - int(
            raw["after_plan"]["selected_parameter_bytes_per_forward"]
        )
        kv_change = int(raw["after_kv_physical"]["full"]["gpu_bytes"]) - int(
            raw["before_kv_physical"]["full"]["gpu_bytes"]
        )
        changes = {
            "Expert GPU pages": expert_change,
            "KV GPU pages": kv_change,
            "Net accounted pages": expert_change + kv_change,
        }
        transition_changes[transition] = changes
        add_component_records(
            records,
            scope="transition",
            order=TRANSITIONS.index(transition),
            transition=transition,
            from_plan=PLAN_NAMES[before_plan_class],
            to_plan=PLAN_NAMES[after_plan_class],
            changes=changes,
            source_sections={
                "Expert GPU pages": (
                    "attempt-03 result.json transitions[*].before_plan/after_plan."
                    "selected_parameter_bytes_per_forward"
                ),
                "KV GPU pages": (
                    "attempt-03 result.json transitions[*].before_kv_physical/"
                    "after_kv_physical.full.gpu_bytes"
                ),
                "Net accounted pages": "derived from Expert GPU pages + KV GPU pages",
            },
        )

    require(tuple(transition_changes) == TRANSITIONS, "transition order differs")
    for outward, inward in (("K2→K4", "K4→K2"), ("K4→K6", "K6→K4")):
        for component in COMPONENTS:
            require(
                transition_changes[outward][component] == -transition_changes[inward][component],
                f"round-trip asymmetry for {outward}/{inward} {component}",
            )

    cumulative_definitions = (
        ("K2→K6 cumulative", "K2", "K6", ("K2→K4", "K4→K6")),
        ("Full round trip", "K2", "K2", TRANSITIONS),
    )
    for cumulative_index, (label, from_plan, to_plan, members) in enumerate(
        cumulative_definitions
    ):
        changes = {
            component: sum(transition_changes[member][component] for member in members)
            for component in COMPONENTS
        }
        add_component_records(
            records,
            scope="cumulative",
            order=cumulative_index,
            transition=label,
            from_plan=from_plan,
            to_plan=to_plan,
            changes=changes,
            source_sections={
                component: f"derived from attempt-03 transitions: {', '.join(members)}"
                for component in COMPONENTS
            },
        )

    data = pd.DataFrame.from_records(records, columns=CSV_COLUMNS)
    require(len(data) == 18, "expected 18 tidy accounting rows")
    require(not data.duplicated(["scope", "transition", "component"]).any(),
            "duplicate accounting row")

    cumulative = data[data["scope"] == "cumulative"].set_index(["transition", "component"])
    require(int(cumulative.loc[("K2→K6 cumulative", "Net accounted pages"), "change_pages"]) == -120,
            "K2→K6 cumulative difference is not 120 pages")
    require(
        (cumulative.loc["Full round trip", "change_pages"] == 0).all(),
        "full round trip does not return to zero",
    )
    return data


def write_and_reload_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.9f")
    serialized = pd.read_csv(path)
    require(list(serialized.columns) == list(CSV_COLUMNS), "CSV columns differ")
    require(len(serialized) == len(data), "serialized row count differs")
    require(
        (serialized["change_bytes"] == serialized["change_pages"] * ACCOUNTING_PAGE_BYTES).all(),
        "serialized page conversion differs",
    )
    return serialized


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.3,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def signed_integer(value: int) -> str:
    return f"{value:+,}".replace("-", "−")


def render_figure(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    configure_matplotlib()
    figure, (transition_axis, cumulative_axis) = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.75),
        gridspec_kw={"width_ratios": (3.20, 1.55)},
    )

    transition_data = data[data["scope"] == "transition"]
    centers = np.arange(len(TRANSITIONS), dtype=float)
    width = 0.235
    offsets = np.array((-width, 0.0, width))

    for component_index, component in enumerate(COMPONENTS):
        panel = transition_data[transition_data["component"] == component].set_index(
            "transition"
        ).loc[list(TRANSITIONS)]
        values = panel["change_pages"].to_numpy(dtype=float)
        bars = transition_axis.bar(
            centers + offsets[component_index],
            values / 1000.0,
            width=width * 0.90,
            color=COLORS[component],
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            hatch=HATCHES[component],
            zorder=3,
        )
        for bar, value in zip(bars, values.astype(int), strict=True):
            plotted_value = value / 1000.0
            text_y = plotted_value + (1.5 if value >= 0 else -1.5)
            transition_axis.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                signed_integer(value),
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=6.8,
                color="#222222",
                clip_on=False,
            )

    transition_axis.axhline(0, color="#555555", linewidth=1.0, zorder=2)
    transition_axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.85, zorder=0)
    transition_axis.set_axisbelow(True)
    transition_axis.set_xticks(centers, TRANSITIONS)
    transition_axis.set_ylim(-41.5, 41.5)
    transition_axis.set_yticks((-40, -20, 0, 20, 40))
    transition_axis.set_ylabel("Change in 64 KiB GPU pages (×10³)")
    transition_axis.set_title("(a) Per-transition accounting", loc="left", pad=4)

    cumulative = data[
        (data["scope"] == "cumulative") & (data["component"] == "Net accounted pages")
    ].sort_values("order")
    cumulative_values = cumulative["change_pages"].to_numpy(dtype=int)
    cumulative_x = np.arange(len(cumulative_values), dtype=float)
    cumulative_axis.vlines(
        cumulative_x,
        0,
        cumulative_values,
        color=COLORS["Net accounted pages"],
        linewidth=2.0,
        zorder=2,
    )
    cumulative_axis.scatter(
        cumulative_x,
        cumulative_values,
        s=30,
        color=COLORS["Net accounted pages"],
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
        zorder=3,
    )
    cumulative_axis.axhline(0, color="#555555", linewidth=1.0, zorder=1)
    cumulative_axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.85, zorder=0)
    cumulative_axis.set_axisbelow(True)
    cumulative_axis.set_xticks(cumulative_x, ("K2→K6", "Full\nround trip"))
    cumulative_axis.set_xlim(-0.25, 1.25)
    cumulative_axis.set_ylim(-165, 45)
    cumulative_axis.set_yticks((-150, -100, -50, 0))
    cumulative_axis.set_ylabel("Net change (64 KiB pages)")
    cumulative_axis.set_title("(b) Cumulative balance", loc="left", pad=4)
    cumulative_axis.text(
        0.08,
        -120,
        "−120 pages\n(−7.5 MiB)",
        ha="left",
        va="center",
        fontsize=7.0,
    )
    cumulative_axis.text(1, 10, "0 pages", ha="center", va="bottom", fontsize=7.0)

    for axis in (transition_axis, cumulative_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=3)
        axis.tick_params(axis="y", length=3, width=0.7)

    figure.legend(
        handles=[
            Patch(
                facecolor=COLORS[component],
                edgecolor=EDGE_COLOR,
                linewidth=0.7,
                hatch=HATCHES[component],
                label=component,
            )
            for component in COMPONENTS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.6,
    )
    figure.subplots_adjust(left=0.095, right=0.975, bottom=0.20, top=0.80, wspace=0.42)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, format="pdf")
    figure.savefig(png_path, format="png", dpi=300)
    plt.close(figure)


def write_caption(path: Path) -> None:
    caption = (
        "Audited GPU-resident physical-page accounting across one "
        "K2→K4→K6→K4→K2 qualification round trip (64 KiB accounting pages). "
        "Expert deltas are obtained from the before/after selected expert bytes, KV "
        "deltas from the audited before/after GPU-resident KV bytes, and the reported "
        "net is their sum. Adjacent transitions are not one-to-one: across K2→K6, "
        "65,664 expert pages are released while 65,544 KV pages are added, leaving "
        "a 120-page (7.5 MiB) difference. The reverse transitions are symmetric, and "
        "the complete round trip returns both accounts to their initial values with zero "
        "net change. All KV snapshots have a GPU-only active prefix, a CPU-only tail, "
        "zero unknown bytes, and an unchanged fixed-VA hash. Simultaneous CPU and GPU "
        "residency after expert restoration reflects the retained immutable CPU shadow "
        "and is not double-counted here. This single qualification round trip accounts "
        "for expert and KV residency but does not independently count shared, runtime, "
        "workspace, or allocator pages; the broader identity remains ΔResident + ΔShared "
        "+ ΔKV + ΔRuntime/Workspace/Allocator = 0."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")


def print_summary(data: pd.DataFrame) -> None:
    for transition in TRANSITIONS:
        panel = data[
            (data["scope"] == "transition") & (data["transition"] == transition)
        ].set_index("component")
        print(
            f"{transition}: expert={signed_integer(int(panel.loc['Expert GPU pages', 'change_pages']))} "
            f"pages, KV={signed_integer(int(panel.loc['KV GPU pages', 'change_pages']))} pages, "
            f"net={signed_integer(int(panel.loc['Net accounted pages', 'change_pages']))} pages"
        )
    cumulative = data[
        (data["scope"] == "cumulative") & (data["component"] == "Net accounted pages")
    ].set_index("transition")
    print(
        "K2→K6 cumulative net: "
        f"{signed_integer(int(cumulative.loc['K2→K6 cumulative', 'change_pages']))} pages"
    )
    print(
        "Full-round-trip net: "
        f"{signed_integer(int(cumulative.loc['Full round trip', 'change_pages']))} pages"
    )


def main() -> None:
    args = parse_args()
    data = build_data(args.result)
    serialized = write_and_reload_csv(data, args.csv)
    render_figure(serialized, args.pdf, args.png)
    write_caption(args.caption)
    print_summary(serialized)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png} (300 dpi)")
    print(f"Wrote {args.caption}")


if __name__ == "__main__":
    main()
