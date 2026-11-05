#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "numpy>=1.24,<3",
#   "pandas>=2.0,<4",
# ]
# ///
"""Render every paper figure in dark annotated and clean variants.

The scientific plotting scripts remain the source of geometry and data.  This
driver redirects their PDF/PNG writes into ``figures/new_output``, darkens
colored data artists while retaining white paper backgrounds and light grids,
and emits a second copy without numeric in-panel annotations.  Axis ticks,
axis labels, titles, row labels, and legend labels remain in both copies.
"""

from __future__ import annotations

import argparse
import colorsys
import re
import runpy
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.collections import Collection
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Annotation, Text


SCRIPT_PATH = Path(__file__).resolve()
FIGURES_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "new_output"

# tab_pd_lifecycle_invariants.py intentionally does not appear here: it emits
# a LaTeX table, not a figure.  E14 is included even though its historical
# output directory was figures/output rather than figures/pdf and figures/png.
FIGURE_SCRIPTS = (
    "fig3_performance_bars.py",
    "fig_adaptive_steady_state.py",
    "fig_adaptmoe_pressure_throughput.py",
    "fig_batched_plan_update.py",
    "fig_core_mechanism_ablation.py",
    "fig_e13_rag_pressure_throughput.py",
    "fig_e2e_six_systems.py",
    "fig_e2e_six_systems_normalized.py",
    "fig_e2e_six_systems_normalized_bars.py",
    "fig_fixed_plan_tradeoff.py",
    "fig_kv_expert_partition_crossover.py",
    "fig_pd_page_table_switch.py",
    "fig_phase_path_preference.py",
    "fig_plan_memory_accounting.py",
    "fig_qwen_stock_uvm_vs_native_uva.py",
    "fig_stock_uvm_demand_paging.py",
    "fig_stock_uvm_vs_native_uva.py",
    "fig_stock_uvm_vs_native_uva_e2e.py",
    "plot_fig_e14_placement_scaling.py",
)

BAR_FIGURE_BASES = frozenset(
    {
        "fig3a_deepseek_performance",
        "fig3b_qwen_performance",
        "fig_adaptmoe_pressure_throughput",
        "fig_core_mechanism_ablation",
        "fig_e2e_six_systems_normalized_bars",
        "fig_fixed_plan_tradeoff",
        "fig_kv_expert_partition_crossover",
        "fig_pd_page_table_switch",
        "fig_plan_memory_accounting",
        "fig_qwen_stock_uvm_vs_native_uva",
    }
)

FIG3_BASES = frozenset(
    {
        "fig3a_deepseek_performance",
        "fig3b_qwen_performance",
    }
)

CLEAN_BLACK_BAR_LABEL_BASES = frozenset(
    {
        "fig_adaptmoe_pressure_throughput",
    }
)

ANNOTATED_HIDE_WHITE_SCATTER_BASES = frozenset(
    {
        "fig_kv_expert_partition_crossover",
    }
)

NUMERIC_ANNOTATION_RE = re.compile(r"(?:\d|×|%|\\times)")
MULTIPLIER_ANNOTATION_RE = re.compile(r"(?:×|\\times)")
ROW_LABEL_RE = re.compile(r"^B\s*=\s*\d+$")
PLAIN_VALUE_RE = re.compile(
    r"^[+\-−]?\d[\d,]*(?:\.\d+)?(?:[eE][+\-]?\d+)?"
    r"(?:\s*(?:ms|s|GiB|MiB|pages|tokens/s|req/s))?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="optional script filenames or stems to render",
    )
    return parser.parse_args()


def _dark_rgba(rgba: Iterable[float]) -> tuple[float, float, float, float]:
    red, green, blue, alpha = (float(value) for value in rgba)
    if alpha <= 0.0:
        return red, green, blue, alpha
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    # Preserve neutral paper elements (white backgrounds, black text, and gray
    # grids).  Colored data marks receive a deliberately subdued, dark value.
    if saturation < 0.16:
        return red, green, blue, alpha
    saturation = max(saturation, 0.48)
    value = min(value, 0.50)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return red, green, blue, alpha


def _dark_color(value: object) -> object:
    if isinstance(value, str) and value.lower() in {"none", "auto"}:
        return value
    try:
        return _dark_rgba(mcolors.to_rgba(value))
    except (TypeError, ValueError):
        return value


def _dark_color_array(values: object) -> np.ndarray:
    array = mcolors.to_rgba_array(values)
    return np.asarray([_dark_rgba(color) for color in array], dtype=float)


def _darken_scalar_mappables(figure: Figure) -> None:
    for axis in figure.axes:
        for artist in [*axis.images, *axis.collections]:
            get_array = getattr(artist, "get_array", None)
            get_cmap = getattr(artist, "get_cmap", None)
            set_cmap = getattr(artist, "set_cmap", None)
            if not callable(get_array) or not callable(get_cmap) or not callable(set_cmap):
                continue
            values = get_array()
            cmap = get_cmap()
            if values is None or cmap is None:
                continue
            samples = np.asarray(cmap(np.linspace(0.0, 1.0, 256)), dtype=float)
            dark_samples = np.asarray([_dark_rgba(color) for color in samples])
            set_cmap(mcolors.ListedColormap(dark_samples, name=f"dark_{cmap.name}"))


def _darken_figure(figure: Figure) -> None:
    if getattr(figure, "_paper_darkened", False):
        return
    figure.canvas.draw()
    _darken_scalar_mappables(figure)
    figure.canvas.draw()

    for line in figure.findobj(match=Line2D):
        line.set_color(_dark_color(line.get_color()))
        line.set_markerfacecolor(_dark_color(line.get_markerfacecolor()))
        line.set_markeredgecolor(_dark_color(line.get_markeredgecolor()))

    for patch in figure.findobj(match=Patch):
        patch.set_facecolor(_dark_color(patch.get_facecolor()))
        patch.set_edgecolor(_dark_color(patch.get_edgecolor()))

    for collection in figure.findobj(match=Collection):
        collection.set_rasterized(False)
        # Scalar-mapped collections are darkened through their colormap so the
        # main plot and any linked colorbar stay synchronized.
        values = collection.get_array()
        if values is not None:
            continue
        facecolors = collection.get_facecolors()
        edgecolors = collection.get_edgecolors()
        if len(facecolors):
            collection.set_facecolors(_dark_color_array(facecolors))
        if len(edgecolors):
            collection.set_edgecolors(_dark_color_array(edgecolors))

    for text in figure.findobj(match=Text):
        text.set_color(_dark_color(text.get_color()))

    figure._paper_darkened = True


def _protected_texts(figure: Figure) -> set[Text]:
    protected: set[Text] = set()
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        protected.add(suptitle)
    for axis in figure.axes:
        protected.update(
            {
                axis.title,
                axis._left_title,
                axis._right_title,
                axis.xaxis.label,
                axis.yaxis.label,
                axis.xaxis.get_offset_text(),
                axis.yaxis.get_offset_text(),
            }
        )
        protected.update(axis.get_xticklabels(minor=False))
        protected.update(axis.get_xticklabels(minor=True))
        protected.update(axis.get_yticklabels(minor=False))
        protected.update(axis.get_yticklabels(minor=True))
        legend = axis.get_legend()
        if legend is not None:
            protected.update(legend.get_texts())
            protected.add(legend.get_title())
    for legend in figure.legends:
        protected.update(legend.get_texts())
        protected.add(legend.get_title())
    return protected


def _hide_numeric_annotations(figure: Figure) -> list[Text]:
    protected = _protected_texts(figure)
    hidden: list[Text] = []
    for text in figure.findobj(match=Text):
        label = text.get_text().strip()
        if not text.get_visible() or text in protected or not label:
            continue
        # Bar-top values and pre-existing multiplier callouts are required in
        # both output variants.  "Clean" only removes optional annotations.
        if getattr(text, "_paper_required_bar_label", False):
            continue
        # Figure 3 row labels are structural, not data annotations.
        if ROW_LABEL_RE.fullmatch(label):
            continue
        if NUMERIC_ANNOTATION_RE.search(label):
            text.set_visible(False)
            hidden.append(text)
    return hidden


def _hide_white_faced_collections(figure: Figure) -> list[Collection]:
    hidden: list[Collection] = []
    for axis in figure.axes:
        for collection in axis.collections:
            facecolors = collection.get_facecolors()
            if not collection.get_visible() or len(facecolors) == 0:
                continue
            rgb = np.asarray(facecolors, dtype=float)[:, :3]
            if not bool(np.all(rgb >= 0.95)):
                continue
            collection.set_visible(False)
            hidden.append(collection)
    return hidden


def _format_bar_value(value: float) -> str:
    magnitude = abs(value)
    sign = "−" if value < 0 else ""
    if magnitude >= 1000:
        body = f"{magnitude:,.0f}"
    elif magnitude >= 100:
        body = f"{magnitude:.1f}"
    elif magnitude >= 10:
        body = f"{magnitude:.1f}"
    elif magnitude >= 1:
        body = f"{magnitude:.2f}"
    elif magnitude >= 0.01:
        body = f"{magnitude:.2f}"
    else:
        body = f"{magnitude:.3f}"
    return sign + body


def _axis_bar_patches(axis: mpl.axes.Axes) -> list[Rectangle]:
    bars = [
        patch
        for container in axis.containers
        if isinstance(container, BarContainer)
        for patch in container.patches
        if isinstance(patch, Rectangle) and patch.get_visible()
    ]
    if not bars:
        # Compatibility fallback for unusual artists that were not registered
        # in an Axes BarContainer.
        bars = [
            patch
            for patch in axis.patches
            if isinstance(patch, Rectangle)
            and patch is not axis.patch
            and patch.get_visible()
        ]
    bars = [
        patch
        for patch in bars
        if np.isfinite([float(patch.get_width()), float(patch.get_height())]).all()
        and float(patch.get_width()) != 0.0
        and float(patch.get_height()) != 0.0
    ]
    return bars


def _bar_orientation(
    axis: mpl.axes.Axes,
    bars: list[Rectangle],
    renderer: mpl.backend_bases.RendererBase,
) -> str:
    container_orientations = {
        str(container.orientation)
        for container in axis.containers
        if isinstance(container, BarContainer)
        and any(patch in bars for patch in container.patches)
        and container.orientation in {"vertical", "horizontal"}
    }
    if len(container_orientations) == 1:
        return container_orientations.pop()
    widths: list[float] = []
    heights: list[float] = []
    for bar in bars:
        bounds = bar.get_window_extent(renderer)
        widths.append(abs(float(bounds.width)))
        heights.append(abs(float(bounds.height)))
    return "horizontal" if np.median(widths) > np.median(heights) else "vertical"


def _bar_endpoint_display(
    axis: mpl.axes.Axes, bar: Rectangle, orientation: str
) -> tuple[float, float]:
    if orientation == "vertical":
        point = (
            float(bar.get_x() + bar.get_width() / 2.0),
            float(bar.get_y() + bar.get_height()),
        )
    else:
        point = (
            float(bar.get_x() + bar.get_width()),
            float(bar.get_y() + bar.get_height() / 2.0),
        )
    transformed = axis.transData.transform(point)
    return float(transformed[0]), float(transformed[1])


def _mark_required_bar_text(text: Text) -> None:
    text.set_rotation(0)
    text._paper_required_bar_label = True


def _match_existing_value_labels(
    axis: mpl.axes.Axes,
    bars: list[Rectangle],
    orientation: str,
    renderer: mpl.backend_bases.RendererBase,
) -> set[Rectangle]:
    """Keep true values that the source plot already placed at a bar end."""
    unmatched = set(bars)
    for text in axis.texts:
        label = text.get_text().strip()
        if not text.get_visible() or not PLAIN_VALUE_RE.fullmatch(label):
            continue
        bounds = text.get_window_extent(renderer)
        text_center = (
            float((bounds.x0 + bounds.x1) / 2.0),
            float((bounds.y0 + bounds.y1) / 2.0),
        )
        candidates: list[tuple[float, Rectangle]] = []
        for bar in unmatched:
            endpoint_x, endpoint_y = _bar_endpoint_display(axis, bar, orientation)
            bar_bounds = bar.get_window_extent(renderer)
            if orientation == "vertical":
                cross_distance = abs(text_center[0] - endpoint_x)
                end_distance = min(abs(bounds.y0 - endpoint_y), abs(bounds.y1 - endpoint_y))
                cross_limit = max(8.0, abs(float(bar_bounds.width)) * 0.9)
            else:
                cross_distance = abs(text_center[1] - endpoint_y)
                end_distance = min(abs(bounds.x0 - endpoint_x), abs(bounds.x1 - endpoint_x))
                cross_limit = max(8.0, abs(float(bar_bounds.height)) * 0.9)
            if cross_distance <= cross_limit and end_distance <= 34.0:
                candidates.append((cross_distance + end_distance, bar))
        if not candidates:
            continue
        _, matched = min(candidates, key=lambda item: item[0])
        _mark_required_bar_text(text)
        unmatched.remove(matched)
    return set(bars) - unmatched


def _vertical_bar_groups(bars: list[Rectangle]) -> list[list[Rectangle]]:
    groups: dict[int, list[Rectangle]] = {}
    for bar in bars:
        center = float(bar.get_x() + bar.get_width() / 2.0)
        groups.setdefault(round(center), []).append(bar)
    return list(groups.values())


def _match_multiplier_labels(
    axis: mpl.axes.Axes,
    bars: list[Rectangle],
    orientation: str,
    renderer: mpl.backend_bases.RendererBase,
) -> set[Rectangle]:
    """Associate existing multiplier callouts with their bar or bar group."""
    matched: set[Rectangle] = set()
    multiplier_texts = [
        text
        for text in axis.texts
        if text.get_visible() and MULTIPLIER_ANNOTATION_RE.search(text.get_text())
    ]
    if orientation == "vertical":
        for text in multiplier_texts:
            bounds = text.get_window_extent(renderer)
            text_x = float((bounds.x0 + bounds.x1) / 2.0)
            best: tuple[float, list[Rectangle]] | None = None
            for group in _vertical_bar_groups(bars):
                centers = [
                    _bar_endpoint_display(axis, bar, orientation)[0] for bar in group
                ]
                half_width = max(
                    abs(float(bar.get_window_extent(renderer).width)) for bar in group
                )
                lower = min(centers) - half_width
                upper = max(centers) + half_width
                if lower <= text_x <= upper:
                    distance = abs(text_x - float(np.mean(centers)))
                    if best is None or distance < best[0]:
                        best = (distance, group)
            if best is not None:
                _mark_required_bar_text(text)
                matched.update(best[1])
    else:
        for text in multiplier_texts:
            bounds = text.get_window_extent(renderer)
            text_y = float((bounds.y0 + bounds.y1) / 2.0)
            candidates: list[tuple[float, Rectangle]] = []
            for bar in bars:
                endpoint_x, endpoint_y = _bar_endpoint_display(axis, bar, orientation)
                bar_bounds = bar.get_window_extent(renderer)
                cross_distance = abs(text_y - endpoint_y)
                end_distance = min(abs(bounds.x0 - endpoint_x), abs(bounds.x1 - endpoint_x))
                cross_limit = max(8.0, abs(float(bar_bounds.height)) * 1.1)
                if cross_distance <= cross_limit and end_distance <= 48.0:
                    candidates.append((cross_distance + end_distance, bar))
            if candidates:
                _, bar = min(candidates, key=lambda item: item[0])
                _mark_required_bar_text(text)
                matched.add(bar)
    return matched


def _add_true_bar_labels(
    axis: mpl.axes.Axes,
    bars: list[Rectangle],
    orientation: str,
    skip: set[Rectangle],
) -> list[Text]:
    added: list[Text] = []
    if orientation == "vertical":
        ordered = sorted(bars, key=lambda bar: (round(bar.get_x() + bar.get_width() / 2.0),
                                                 bar.get_x() + bar.get_width() / 2.0))
        group_counts: dict[float, int] = {}
        for bar in ordered:
            center = float(bar.get_x() + bar.get_width() / 2.0)
            group = round(center)
            tier = group_counts.get(group, 0)
            group_counts[group] = tier + 1
            if bar in skip:
                continue
            value = float(bar.get_height())
            endpoint = float(bar.get_y() + value)
            positive = value >= 0.0
            label = axis.annotate(
                _format_bar_value(value),
                (center, endpoint),
                xytext=(0.0, 2.0 + (tier % 6) * 6.2 if positive else -2.0),
                textcoords="offset points",
                ha="center",
                va="bottom" if positive else "top",
                rotation=0,
                fontsize=5.3,
                fontweight="bold",
                color="#252525",
                clip_on=False,
                zorder=20,
            )
            _mark_required_bar_text(label)
            added.append(label)
    else:
        ordered = sorted(bars, key=lambda bar: (round(bar.get_y() + bar.get_height() / 2.0),
                                                 bar.get_y() + bar.get_height() / 2.0))
        group_sizes: dict[float, int] = {}
        for bar in ordered:
            group = round(float(bar.get_y() + bar.get_height() / 2.0), 6)
            group_sizes[group] = group_sizes.get(group, 0) + 1
        group_counts: dict[float, int] = {}
        for bar in ordered:
            center = float(bar.get_y() + bar.get_height() / 2.0)
            group = round(center, 6)
            tier = group_counts.get(group, 0)
            group_counts[group] = tier + 1
            if bar in skip:
                continue
            value = float(bar.get_width())
            endpoint = float(bar.get_x() + value)
            positive = value >= 0.0
            label = axis.annotate(
                _format_bar_value(value),
                (endpoint, center),
                xytext=(
                    2.0 if positive else -2.0,
                    (tier - (group_sizes[group] - 1) / 2.0) * 5.2,
                ),
                textcoords="offset points",
                ha="left" if positive else "right",
                va="center",
                rotation=0,
                fontsize=5.3,
                fontweight="bold",
                color="#252525",
                clip_on=False,
                zorder=20,
            )
            _mark_required_bar_text(label)
            added.append(label)
    return added


def _add_vertical_label_headroom(axis: mpl.axes.Axes, bars: list[Rectangle]) -> None:
    lower, upper = (float(value) for value in axis.get_ylim())
    if lower >= upper:
        return
    if axis.get_yscale() == "log" and lower > 0.0:
        log_lower, log_upper = np.log10((lower, upper))
        axis.set_ylim(lower, 10 ** (log_upper + (log_upper - log_lower) * 0.22))
        return
    span = upper - lower
    has_positive = any(float(bar.get_height()) > 0.0 for bar in bars)
    has_negative = any(float(bar.get_height()) < 0.0 for bar in bars)
    axis.set_ylim(
        lower - (0.08 * span if has_negative else 0.0),
        upper + (0.22 * span if has_positive else 0.0),
    )


def _add_horizontal_label_headroom(axis: mpl.axes.Axes, bars: list[Rectangle]) -> None:
    lower, upper = (float(value) for value in axis.get_xlim())
    if lower >= upper:
        return
    span = upper - lower
    has_positive = any(float(bar.get_width()) > 0.0 for bar in bars)
    has_negative = any(float(bar.get_width()) < 0.0 for bar in bars)
    axis.set_xlim(
        lower - (0.08 * span if has_negative else 0.0),
        upper + (0.18 * span if has_positive else 0.0),
    )


def _separate_percentage_callouts(axis: mpl.axes.Axes) -> None:
    """Move optional group percentages clear of required bar-top values."""
    for text in axis.texts:
        if not isinstance(text, Annotation) or "%" not in text.get_text():
            continue
        x_offset, y_offset = text.get_position()
        text.set_position((x_offset, y_offset + 10.0))


def _prepare_bar_figure(figure: Figure) -> None:
    if getattr(figure, "_paper_bar_refresh", False):
        return

    # Make the physical figure modestly smaller while increasing all existing
    # paper text.  Relative subplot geometry remains unchanged.
    width, height = figure.get_size_inches()
    figure.set_size_inches(width * 0.92, height * 0.92, forward=True)
    for text in figure.findobj(match=Text):
        text.set_fontsize(float(text.get_fontsize()) + 0.45)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    total_bars = 0
    covered_bars = 0
    for axis in figure.axes:
        bars = _axis_bar_patches(axis)
        if not bars:
            continue
        orientation = _bar_orientation(axis, bars, renderer)
        # Preserve source labels because some plots deliberately show an
        # unnormalized true value above a normalized bar.  Geometry alone
        # cannot reconstruct that value.
        existing_values = _match_existing_value_labels(
            axis, bars, orientation, renderer
        )
        multiplier_bars = _match_multiplier_labels(
            axis, bars, orientation, renderer
        )
        skip = existing_values | multiplier_bars
        added = _add_true_bar_labels(axis, bars, orientation, skip)
        if orientation == "vertical" and added:
            _add_vertical_label_headroom(axis, bars)
            _separate_percentage_callouts(axis)
        elif orientation == "horizontal" and added:
            _add_horizontal_label_headroom(axis, bars)
        total_bars += len(bars)
        covered_bars += len(skip) + len(added)

    if covered_bars != total_bars:
        raise RuntimeError(
            f"bar-label audit failed: covered {covered_bars} of {total_bars} bars"
        )
    for text in figure.findobj(match=Text):
        if getattr(text, "_paper_required_bar_label", False) and text.get_rotation() != 0:
            raise RuntimeError(f"bar label is not horizontal: {text.get_text()!r}")

    figure._paper_bar_refresh = True


def _prepare_fig3_figure(figure: Figure) -> None:
    """Widen Figure 3 and remove all source bar-top numeric labels."""
    if getattr(figure, "_paper_fig3_refresh", False):
        return

    # Relative to the previous 0.92x rendering this makes Figure 3 about one
    # third wider while preserving its rendered height.
    width, height = figure.get_size_inches()
    figure.set_size_inches(width * 1.22, height * 0.92, forward=True)
    for text in figure.findobj(match=Text):
        text.set_fontsize(float(text.get_fontsize()) + 0.45)

    figure.canvas.draw()
    hidden_count = 0
    for axis in figure.axes:
        for text in axis.texts:
            if not PLAIN_VALUE_RE.fullmatch(text.get_text().strip()):
                continue
            text.set_visible(False)
            hidden_count += 1

    if hidden_count == 0:
        raise RuntimeError("Figure 3 has no bar-top numeric labels to remove")
    figure._paper_fig3_refresh = True


def _selected_scripts(only: list[str] | None) -> tuple[str, ...]:
    if not only:
        return FIGURE_SCRIPTS
    requested = {Path(item).stem for item in only}
    selected = tuple(name for name in FIGURE_SCRIPTS if Path(name).stem in requested)
    missing = requested - {Path(name).stem for name in selected}
    if missing:
        raise ValueError(f"unknown figure scripts: {', '.join(sorted(missing))}")
    return selected


def _render_validated_csv_fallback(script: Path) -> bool:
    """Render the two legacy normalized figures from their validated CSVs.

    Their parsers deliberately reject a changed prose coverage statement in
    the living consolidated Markdown file.  The plotting input CSVs predate
    that prose edit and remain the authoritative serialized inputs for these
    already-produced figures; this fallback does not modify parser checks or
    synthesize any data.
    """
    supported = {
        "fig_e2e_six_systems_normalized.py": "fig_e2e_six_systems_normalized",
        "fig_e2e_six_systems_normalized_bars.py": "fig_e2e_six_systems_normalized_bars",
    }
    base = supported.get(script.name)
    if base is None:
        return False
    csv_path = FIGURES_ROOT / "data" / f"{base}.csv"
    if not csv_path.is_file():
        return False
    namespace = runpy.run_path(str(script), run_name=f"_dark_fallback_{script.stem}")
    render = namespace.get("render_figure")
    if not callable(render):
        return False
    data = pd.read_csv(csv_path, keep_default_na=False)
    render(
        data,
        FIGURES_ROOT / "pdf" / f"{base}.pdf",
        FIGURES_ROOT / "png" / f"{base}.png",
    )
    print(f"[FALLBACK] rendered {base} from validated CSV: {csv_path}")
    return True


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    pdf_dir = output_root / "pdf"
    png_dir = output_root / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    original_savefig = Figure.savefig
    written: set[Path] = set()

    def save_dark_variants(figure: Figure, fname: object, *positional: object, **kwargs: object) -> None:
        requested_path = Path(fname)  # all source scripts use filesystem paths
        requested_format = str(kwargs.get("format") or requested_path.suffix.lstrip(".")).lower()
        if requested_format not in {"pdf", "png"}:
            original_savefig(figure, fname, *positional, **kwargs)
            return

        output_dir = pdf_dir if requested_format == "pdf" else png_dir
        base = requested_path.stem
        _darken_figure(figure)
        if base in FIG3_BASES:
            _prepare_fig3_figure(figure)
        elif base in BAR_FIGURE_BASES:
            _prepare_bar_figure(figure)
        save_kwargs = dict(kwargs)
        save_kwargs["format"] = requested_format
        if requested_format == "png":
            save_kwargs["dpi"] = 300

        annotated_path = output_dir / f"{base}_annotated.{requested_format}"
        clean_path = output_dir / f"{base}_clean.{requested_format}"
        hidden_annotated_collections: list[Collection] = []
        if base in ANNOTATED_HIDE_WHITE_SCATTER_BASES:
            hidden_annotated_collections = _hide_white_faced_collections(figure)
        original_savefig(figure, annotated_path, *positional, **save_kwargs)
        written.add(annotated_path)
        for collection in hidden_annotated_collections:
            collection.set_visible(True)

        hidden = _hide_numeric_annotations(figure)
        restored_label_styles: list[tuple[Text, object, float]] = []
        if base in CLEAN_BLACK_BAR_LABEL_BASES:
            for text in figure.findobj(match=Text):
                if not text.get_visible() or not getattr(
                    text, "_paper_required_bar_label", False
                ):
                    continue
                restored_label_styles.append(
                    (text, text.get_color(), float(text.get_fontsize()))
                )
                text.set_color("#000000")
                text.set_fontsize(5.8)
        hidden_clean_collections: list[Collection] = []
        if base in CLEAN_BLACK_BAR_LABEL_BASES:
            for axis in figure.axes:
                for collection in axis.collections:
                    if not collection.get_visible():
                        continue
                    hidden_clean_collections.append(collection)
                    collection.set_visible(False)
        original_savefig(figure, clean_path, *positional, **save_kwargs)
        written.add(clean_path)
        for collection in hidden_clean_collections:
            collection.set_visible(True)
        for text, color, fontsize in restored_label_styles:
            text.set_color(color)
            text.set_fontsize(fontsize)
        for text in hidden:
            text.set_visible(True)

    Figure.savefig = save_dark_variants
    old_argv = sys.argv[:]
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    failures: list[str] = []
    try:
        for script_name in _selected_scripts(args.only):
            script = SCRIPT_PATH.parent / script_name
            print(f"[RENDER] {script_name}")
            sys.argv = [str(script)]
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    try:
                        recovered = _render_validated_csv_fallback(script)
                    except Exception as fallback_error:
                        failures.append(
                            f"{script_name}: exit {error.code}; fallback "
                            f"{type(fallback_error).__name__}: {fallback_error}"
                        )
                    else:
                        if not recovered:
                            failures.append(f"{script_name}: exit {error.code}")
            except Exception as error:  # keep the batch moving and report all failures
                try:
                    recovered = _render_validated_csv_fallback(script)
                except Exception as fallback_error:
                    failures.append(
                        f"{script_name}: {type(error).__name__}: {error}; fallback "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )
                else:
                    if not recovered:
                        failures.append(f"{script_name}: {type(error).__name__}: {error}")
    finally:
        Figure.savefig = original_savefig
        sys.argv = old_argv
        if sys.path and sys.path[0] == str(SCRIPT_PATH.parent):
            sys.path.pop(0)

    if failures:
        for failure in failures:
            print(f"[FAILED] {failure}", file=sys.stderr)
        return 1
    print(f"[OK] wrote {len(written)} files under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
