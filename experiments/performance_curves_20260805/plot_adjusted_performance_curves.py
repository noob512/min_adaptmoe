#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 1125
SCALE = 2
BG = "#FFFFFF"
TEXT = "#17202A"
MUTED = "#5F6B76"
GRID = "#D9DEE3"
AXIS = "#39434C"
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
CJK_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path(
        "/tmp/nvidia_uvm_curve_fonts/usr/share/fonts/truetype/"
        "wqy/wqy-microhei.ttc"
    ),
)


@dataclass(frozen=True)
class Record:
    figure: str
    concurrency: int
    series: str
    throughput: float
    status: str
    stddev: float | None


@dataclass(frozen=True)
class PlotArea:
    left: int = 190
    top: int = 225
    right: int = 1715
    bottom: int = 920

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def font(
    size: int,
    bold: bool = False,
    cjk: bool = False,
) -> ImageFont.FreeTypeFont:
    if cjk:
        configured = os.environ.get("NVIDIA_UVM_CJK_FONT")
        candidates = ((Path(configured),) if configured else ()) + CJK_FONT_CANDIDATES
        for candidate in candidates:
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size * SCALE)
        raise FileNotFoundError(
            "Chinese font not found; set NVIDIA_UVM_CJK_FONT to a CJK font file"
        )
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_ROOT / name), size * SCALE)


def read_records(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_stddev = (row.get("stddev_tok_s") or "").strip()
            records.append(
                Record(
                    figure=row["figure"],
                    concurrency=int(row["concurrency"]),
                    series=row["series"],
                    throughput=float(row["throughput_tok_s"]),
                    status=row["status"],
                    stddev=float(raw_stddev) if raw_stddev else None,
                )
            )
    return records


def xy_mapper(
    area: PlotArea,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    def map_xy(x: float, y: float) -> tuple[int, int]:
        px = area.left + (x - x_min) / (x_max - x_min) * area.width
        py = area.bottom - (y - y_min) / (y_max - y_min) * area.height
        return round(px * SCALE), round(py * SCALE)

    return map_xy


def text_size(draw: ImageDraw.ImageDraw, value: str, text_font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), value, font=text_font)
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    text_font,
    fill: str = TEXT,
    anchor: str = "la",
) -> None:
    draw.text(
        (round(xy[0] * SCALE), round(xy[1] * SCALE)),
        value,
        font=text_font,
        fill=fill,
        anchor=anchor,
    )


def draw_vertical_label(
    image: Image.Image,
    value: str,
    center: tuple[int, int],
    text_font,
) -> None:
    probe = ImageDraw.Draw(image)
    width, height = text_size(probe, value, text_font)
    layer = Image.new("RGBA", (width + 24 * SCALE, height + 24 * SCALE), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((12 * SCALE, 8 * SCALE), value, font=text_font, fill=TEXT)
    rotated = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(
        rotated,
        (
            round(center[0] * SCALE - rotated.width / 2),
            round(center[1] * SCALE - rotated.height / 2),
        ),
    )


def draw_axes(
    image: Image.Image,
    area: PlotArea,
    x_ticks: Iterable[int],
    y_ticks: Iterable[int],
    y_max: int,
    y_label: str,
    y_min: int = 0,
    tick_font_size: int = 25,
    axis_font_size: int = 30,
    x_label: str = "Concurrent requests",
    cjk_axis_labels: bool = False,
    show_axis_labels: bool = True,
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    x_values = list(x_ticks)
    y_values = list(y_ticks)
    mapped_x_min = min(x_values) if x_min is None else x_min
    mapped_x_max = max(x_values) if x_max is None else x_max
    map_xy = xy_mapper(area, mapped_x_min, mapped_x_max, y_min, y_max)
    tick_font = font(tick_font_size)

    for value in y_values:
        x0, y = map_xy(mapped_x_min, value)
        x1, _ = map_xy(mapped_x_max, value)
        draw.line((x0, y, x1, y), fill=GRID, width=2 * SCALE)
        draw_text(draw, (area.left - 22, y / SCALE), f"{value:,}", tick_font, MUTED, "ra")

    draw.line(
        (
            area.left * SCALE,
            area.top * SCALE,
            area.left * SCALE,
            area.bottom * SCALE,
            area.right * SCALE,
            area.bottom * SCALE,
        ),
        fill=AXIS,
        width=3 * SCALE,
        joint="curve",
    )

    for value in x_values:
        x, y = map_xy(value, y_min)
        draw.line((x, y, x, y + 10 * SCALE), fill=AXIS, width=3 * SCALE)
        draw_text(draw, (x / SCALE, area.bottom + 24), str(value), tick_font, MUTED, "ma")

    if show_axis_labels:
        draw_text(
            draw,
            ((area.left + area.right) / 2, area.bottom + 83),
            x_label,
            font(axis_font_size, bold=True, cjk=cjk_axis_labels),
            TEXT,
            "ma",
        )
        draw_vertical_label(
            image,
            y_label,
            (52, round((area.top + area.bottom) / 2)),
            font(axis_font_size - 1, bold=True, cjk=cjk_axis_labels),
        )


def draw_circle(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    color: str,
    filled: bool,
    width: int = 4,
) -> None:
    r = radius * SCALE
    x, y = center
    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        fill=color if filled else BG,
        outline=color,
        width=width * SCALE,
    )


def draw_diamond(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    color: str,
) -> None:
    x, y = center
    r = radius * SCALE
    draw.polygon(((x, y - r), (x + r, y), (x, y + r), (x - r, y)), fill=color)
    draw.line(
        ((x, y - r), (x + r, y), (x, y + r), (x - r, y), (x, y - r)),
        fill=BG,
        width=2 * SCALE,
        joint="curve",
    )


def draw_series(
    draw: ImageDraw.ImageDraw,
    records: list[Record],
    map_xy,
    color: str,
    line_width: int,
    adaptive: bool = False,
    fill_all: bool = False,
) -> None:
    points = [map_xy(item.concurrency, item.throughput) for item in records]
    draw.line(points, fill=color, width=line_width * SCALE, joint="curve")
    for item, point in zip(records, points):
        if adaptive:
            draw_diamond(draw, point, 10, color)
        else:
            filled = fill_all or item.status in {"measured_mean", "historical_anchor"}
            draw_circle(draw, point, 9 if filled else 8, color, filled)


def draw_grouped_bars(
    draw: ImageDraw.ImageDraw,
    records_by_series: dict[str, list[Record]],
    series_order: list[str],
    map_xy,
    colors: dict[str, str],
    y_baseline: float,
    show_values: bool = False,
    value_font_size: int = 24,
) -> None:
    bar_width = 72 * SCALE
    bar_gap = 18 * SCALE
    series_count = len(series_order)
    for series_index, name in enumerate(series_order):
        offset = (
            series_index - (series_count - 1) / 2
        ) * (bar_width + bar_gap)
        for item in records_by_series[name]:
            center_x, value_y = map_xy(item.concurrency, item.throughput)
            _, baseline_y = map_xy(item.concurrency, y_baseline)
            left = round(center_x + offset - bar_width / 2)
            right = round(center_x + offset + bar_width / 2)
            draw.rectangle(
                (left, min(value_y, baseline_y), right, max(value_y, baseline_y)),
                fill=colors[name],
            )
            if show_values:
                draw_text(
                    draw,
                    (center_x / SCALE + offset / SCALE, value_y / SCALE - 12),
                    f"{item.throughput:.1f}",
                    font(value_font_size, bold=True),
                    TEXT,
                    "mb",
                )


def draw_legend_item(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    color: str,
    adaptive: bool = False,
    font_size: int = 24,
) -> int:
    draw.line(
        (x * SCALE, y * SCALE, (x + 52) * SCALE, y * SCALE),
        fill=color,
        width=(7 if adaptive else 5) * SCALE,
    )
    center = ((x + 26) * SCALE, y * SCALE)
    if adaptive:
        draw_diamond(draw, center, 8, color)
    else:
        draw_circle(draw, center, 7, color, True, width=3)
    legend_font = font(font_size, bold=adaptive)
    draw_text(draw, (x + 66, y), label, legend_font, TEXT, "lm")
    width, _ = text_size(draw, label, legend_font)
    return x + 66 + round(width / SCALE) + 44


def draw_bar_legend_item(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    color: str,
    font_size: int = 24,
) -> int:
    draw.rectangle(
        (
            x * SCALE,
            (y - 11) * SCALE,
            (x + 42) * SCALE,
            (y + 11) * SCALE,
        ),
        fill=color,
    )
    legend_font = font(font_size)
    draw_text(draw, (x + 58, y), label, legend_font, TEXT, "lm")
    width, _ = text_size(draw, label, legend_font)
    return x + 58 + round(width / SCALE) + 44


def save_outputs(image: Image.Image, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    Image.init()
    rgb = image.convert("RGB")
    rgb.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(
        output_base.with_suffix(".png"),
        dpi=(180, 180),
        optimize=True,
    )
    rgb.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(
        output_base.with_suffix(".pdf"),
        resolution=180,
    )


def plot_mechanism_compare(records: list[Record], output_dir: Path) -> None:
    image = Image.new("RGBA", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    draw = ImageDraw.Draw(image)
    area = PlotArea(left=135, top=155, right=1740, bottom=985)
    x_ticks = [32, 64, 96, 128, 160]
    x_min = 16
    x_max = 176
    y_min = 0
    y_max = 350
    map_xy = xy_mapper(area, x_min, x_max, y_min, y_max)

    draw_axes(
        image,
        area,
        x_ticks,
        range(0, y_max + 1, 50),
        y_max,
        "有效吞吐量（tokens/s）",
        y_min=y_min,
        tick_font_size=30,
        axis_font_size=35,
        x_label="并发请求数",
        cjk_axis_labels=True,
        show_axis_labels=False,
        x_min=x_min,
        x_max=x_max,
    )

    styles = {
        "Our adaptive UVM": "#167D8D",
        "MoE-Infinity": "#D46A2C",
    }
    legend_labels = {
        "Our adaptive UVM": "Adaptive UVM (7 GB offload)",
        "MoE-Infinity": "MoE-Infinity (offload + prefetch)",
    }
    by_series = {
        name: sorted(
            [item for item in records if item.series == name],
            key=lambda item: item.concurrency,
        )
        for name in styles
    }
    draw_grouped_bars(
        draw,
        by_series,
        list(styles),
        map_xy,
        styles,
        y_min,
        show_values=True,
        value_font_size=24,
    )

    legend_x = 310
    legend_y = 82
    for name, color in styles.items():
        legend_x = draw_bar_legend_item(
            draw,
            legend_x,
            legend_y,
            legend_labels[name],
            color,
            28,
        )
    save_outputs(image, output_dir / "figure1_ours_vs_moe_infinity")


def plot_offload_scaling(records: list[Record], output_dir: Path) -> None:
    image = Image.new("RGBA", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    draw = ImageDraw.Draw(image)
    area = PlotArea(left=135, top=155, right=1740, bottom=985)
    x_ticks = [64, 96, 128, 160]
    y_min = 400
    y_max = 1400
    map_xy = xy_mapper(area, min(x_ticks), max(x_ticks), y_min, y_max)

    draw_axes(
        image,
        area,
        x_ticks,
        range(y_min, y_max + 1, 200),
        y_max,
        "有效吞吐量（tokens/s）",
        y_min=y_min,
        tick_font_size=30,
        axis_font_size=35,
        x_label="并发请求数",
        cjk_axis_labels=True,
        show_axis_labels=False,
    )

    styles = {
        "Static 4 GB": "#21867A",
        "Static 7 GB": "#D8861E",
        "Static 10 GB": "#C64B4B",
        "Adaptive scaling": "#222B38",
    }
    legend_labels = {
        "Static 4 GB": "4 GB offload",
        "Static 7 GB": "7 GB offload",
        "Static 10 GB": "10 GB offload",
        "Adaptive scaling": "Adaptive",
    }
    by_series = {
        name: sorted(
            [item for item in records if item.series == name],
            key=lambda item: item.concurrency,
        )
        for name in styles
    }
    for name, color in styles.items():
        draw_series(
            draw,
            by_series[name],
            map_xy,
            color,
            9 if name == "Adaptive scaling" else 5,
            adaptive=name == "Adaptive scaling",
        )

    legend_x = 135
    legend_y = 82
    for name, color in styles.items():
        legend_x = draw_legend_item(
            draw,
            legend_x,
            legend_y,
            legend_labels[name],
            color,
            adaptive=name == "Adaptive scaling",
            font_size=28,
        )

    save_outputs(image, output_dir / "figure2_adaptive_offload_scaling")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the adjusted performance curves.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("adjusted_curve_data.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
    )
    args = parser.parse_args()

    records = read_records(args.data)
    plot_mechanism_compare(
        [item for item in records if item.figure == "mechanism_compare"],
        args.output_dir,
    )
    plot_offload_scaling(
        [item for item in records if item.figure == "offload_scaling"],
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
