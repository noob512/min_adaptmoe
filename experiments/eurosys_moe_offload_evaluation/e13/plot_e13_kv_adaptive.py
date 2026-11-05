#!/usr/bin/env python3
"""Plot representative E13 KV pressure, tier, throughput, and latency traces."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import deque
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("qualification", "formal", "capacity"),
        required=True,
    )
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rolling_rate(steps: list[dict[str, Any]], window: int = 8) -> tuple[list[float], list[float]]:
    values: deque[tuple[float, float]] = deque()
    x: list[float] = []
    y: list[float] = []
    for step in steps:
        values.append((float(step["emitted_tokens"]), float(step["step_wall_ms"])))
        while len(values) > window:
            values.popleft()
        elapsed_ms = sum(item[1] for item in values)
        x.append(float(step["elapsed_s"]))
        y.append(sum(item[0] for item in values) * 1000.0 / elapsed_ms if elapsed_ms else 0.0)
    return x, y


def read_nvml(path: Path, trace_start: float, trace_finish: float) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    if not path.is_file():
        return x, y
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                timestamp = float(row["unix_s"])
                if row.get("sample_ok") != "1" or not trace_start <= timestamp <= trace_finish:
                    continue
                x.append(timestamp - trace_start)
                y.append(float(row["gpu_util_percent"]))
            except (KeyError, TypeError, ValueError):
                continue
    return x, y


class Canvas:
    """Small Pillow/SVG renderer so E13 plotting needs no matplotlib install."""

    def __init__(self, width: int, height: int) -> None:
        from PIL import Image, ImageDraw, ImageFont

        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.font = ImageFont.load_default(size=16)
        self.small_font = ImageFont.load_default(size=13)
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]

    def line(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        width: int = 2,
    ) -> None:
        if len(points) < 2:
            return
        self.draw.line(points, fill=fill, width=width, joint="curve")
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(
            f'<polyline points="{encoded}" fill="none" stroke="{fill}" '
            f'stroke-width="{width}" stroke-linejoin="round"/>'
        )

    def rect(
        self,
        box: tuple[float, float, float, float],
        *,
        fill: str | None = None,
        outline: str | None = None,
        width: int = 1,
    ) -> None:
        self.draw.rectangle(box, fill=fill, outline=outline, width=width)
        left, top, right, bottom = box
        self.svg.append(
            f'<rect x="{left:.2f}" y="{top:.2f}" '
            f'width="{right-left:.2f}" height="{bottom-top:.2f}" '
            f'fill="{fill or "none"}" stroke="{outline or "none"}" '
            f'stroke-width="{width}"/>'
        )

    def circle(self, x: float, y: float, radius: int, *, fill: str) -> None:
        self.draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=fill
        )
        self.svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        fill: str = "#202020",
        small: bool = False,
    ) -> None:
        font = self.small_font if small else self.font
        self.draw.text((x, y), value, fill=fill, font=font)
        size = 13 if small else 16
        self.svg.append(
            f'<text x="{x:.2f}" y="{y + size:.2f}" fill="{fill}" '
            f'font-family="sans-serif" font-size="{size}">'
            f'{html.escape(value)}</text>'
        )

    def save(self, png: Path, svg: Path) -> None:
        self.image.save(png)
        svg.write_text("\n".join(self.svg + ["</svg>", ""]), encoding="utf-8")


def scale_points(
    values: list[tuple[float, float]],
    box: tuple[float, float, float, float],
    *,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list[tuple[float, float]]:
    left, top, right, bottom = box
    xspan = max(xmax, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    return [
        (
            left + max(min(x, xmax), 0.0) / xspan * (right - left),
            bottom - (max(min(y, ymax), ymin) - ymin) / yspan * (bottom - top),
        )
        for x, y in values
    ]


def step_values(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not values:
        return []
    stepped = [values[0]]
    previous_y = values[0][1]
    for x, y in values[1:]:
        stepped.append((x, previous_y))
        stepped.append((x, y))
        previous_y = y
    return stepped


def draw_axes(
    canvas: Canvas,
    box: tuple[float, float, float, float],
    *,
    xmax: float,
    ymin: float,
    ymax: float,
    ylabel: str,
    title: str = "",
    log_y: bool = False,
) -> None:
    left, top, right, bottom = box
    canvas.rect(box, outline="#777777")
    if title:
        canvas.text(left, top - 31, title)
    canvas.text(8, top - 27, ylabel, small=True)
    for index in range(5):
        fraction = index / 4
        y = bottom - fraction * (bottom - top)
        canvas.line([(left, y), (right, y)], fill="#e5e5e5", width=1)
        raw = ymin + fraction * (ymax - ymin)
        label = f"{10**raw:.0f}" if log_y else f"{raw:.1f}"
        canvas.text(left - 72, y - 8, label, fill="#555555", small=True)
    for index in range(6):
        fraction = index / 5
        x = left + fraction * (right - left)
        canvas.line([(x, top), (x, bottom)], fill="#f0f0f0", width=1)
        canvas.text(x - 12, bottom + 7, f"{xmax*fraction:.0f}", fill="#555555", small=True)


def main() -> int:
    args = parse_args()
    protocol = load_json(args.config)
    seed = int(protocol["profiles"][args.profile]["seeds"][args.repeat_index])
    repeat_dir = (
        args.run_root.resolve()
        / f"repeat-{args.repeat_index:02d}-seed-{seed}"
    )
    arm_names = [item["name"] for item in protocol["arms"]]
    results = {
        arm: load_json(repeat_dir / arm / "result.json") for arm in arm_names
    }
    if any(result.get("status") != "ok" for result in results.values()):
        raise SystemExit("all representative results must have status=ok")

    colors = {
        "adaptive": "#0072B2",
        "fixed_k2": "#009E73",
        "fixed_k4": "#E69F00",
        "fixed_k6": "#D55E00",
    }
    adaptive = results["adaptive"]
    adaptive_steps = adaptive["time_series"]
    canvas = Canvas(1600, 1360)
    boxes = [
        (120.0, 75.0 + index * 320.0, 1560.0, 330.0 + index * 320.0)
        for index in range(4)
    ]
    xmax = max(
        float(step["elapsed_s"])
        for result in results.values()
        for step in result["time_series"]
    )
    kv_block_bytes = int(protocol["runtime"]["kv_block_bytes"])
    tier_series = {
        arm: [
            (
                float(step["elapsed_s"]),
                int(step["kv"]["current_target_blocks"])
                * kv_block_bytes
                / (1024**3),
            )
            for step in result["time_series"]
        ]
        for arm, result in results.items()
    }
    tier_max = max(y for values in tier_series.values() for _, y in values) * 1.1
    pressure_series = {
        "active blocks": [
            (float(step["elapsed_s"]), int(step["kv"]["active_blocks"] or 0))
            for step in adaptive_steps
        ],
        "predicted blocks": [
            (float(step["elapsed_s"]), int(step["kv"]["predicted_blocks"] or 0))
            for step in adaptive_steps
        ],
        "target blocks": [
            (
                float(step["elapsed_s"]),
                int(step["kv"]["current_target_blocks"] or 0),
            )
            for step in adaptive_steps
        ],
    }
    pressure_max = max(y for values in pressure_series.values() for _, y in values) * 1.1
    rate_series: dict[str, list[tuple[float, float]]] = {}
    for arm, result in results.items():
        rate_x, rate_y = rolling_rate(result["time_series"])
        rate_series[arm] = list(zip(rate_x, rate_y))
    rate_max = max(y for values in rate_series.values() for _, y in values) * 1.1
    latency_series = {
        arm: [
            (float(item["arrival_elapsed_s"]), float(item["e2e_ms"]))
            for item in result["requests"]
            if item.get("success")
        ]
        for arm, result in results.items()
    }
    latency_values = [y for values in latency_series.values() for _, y in values]
    latency_min_log = math.log10(max(min(latency_values) * 0.8, 1.0))
    latency_max_log = math.log10(max(latency_values) * 1.2)

    scenario_order = protocol["profiles"][args.profile]["scenario_names"]
    spans: dict[str, tuple[float, float]] = {}
    for scenario in scenario_order:
        points = [
            float(item["elapsed_s"])
            for item in adaptive_steps
            if item["scenario"] == scenario
        ]
        if points:
            spans[scenario] = (min(points), max(points))
    shade = ["#f5f5f5", "#fff7dc", "#edf7fb", "#fcefee"]
    for index, scenario in enumerate(scenario_order):
        if scenario not in spans:
            continue
        left, right = spans[scenario]
        for box in boxes:
            x0 = box[0] + left / xmax * (box[2] - box[0])
            x1 = box[0] + right / xmax * (box[2] - box[0])
            canvas.rect((x0, box[1], x1, box[3]), fill=shade[index % len(shade)])
        label_x = boxes[0][0] + (left + right) / 2 / xmax * (
            boxes[0][2] - boxes[0][0]
        )
        canvas.text(label_x - 45, boxes[0][1] + 4, scenario, small=True)

    draw_axes(
        canvas,
        boxes[0],
        xmax=xmax,
        ymin=0.0,
        ymax=tier_max,
        ylabel="GPU KV tier (GiB)",
        title="E13 representative cold-process trace",
    )
    draw_axes(
        canvas,
        boxes[1],
        xmax=xmax,
        ymin=0.0,
        ymax=pressure_max,
        ylabel="Adaptive KV blocks",
    )
    draw_axes(
        canvas,
        boxes[2],
        xmax=xmax,
        ymin=0.0,
        ymax=rate_max,
        ylabel="Rolling output tok/s",
    )
    draw_axes(
        canvas,
        boxes[3],
        xmax=xmax,
        ymin=latency_min_log,
        ymax=latency_max_log,
        ylabel="Request E2E ms (log)",
        log_y=True,
    )
    for arm, values in tier_series.items():
        canvas.line(
            scale_points(
                step_values(values),
                boxes[0],
                xmax=xmax,
                ymin=0.0,
                ymax=tier_max,
            ),
            fill=colors[arm],
            width=4 if arm == "adaptive" else 2,
        )
    pressure_colors = {
        "active blocks": "#CC79A7",
        "predicted blocks": "#56B4E9",
        "target blocks": "#0072B2",
    }
    for name, values in pressure_series.items():
        canvas.line(
            scale_points(
                step_values(values) if name == "target blocks" else values,
                boxes[1],
                xmax=xmax,
                ymin=0.0,
                ymax=pressure_max,
            ),
            fill=pressure_colors[name],
            width=2,
        )
    for arm, values in rate_series.items():
        canvas.line(
            scale_points(values, boxes[2], xmax=xmax, ymin=0.0, ymax=rate_max),
            fill=colors[arm],
            width=3 if arm == "adaptive" else 2,
        )
    for arm, values in latency_series.items():
        points = scale_points(
            [(x, math.log10(max(y, 1.0))) for x, y in values],
            boxes[3],
            xmax=xmax,
            ymin=latency_min_log,
            ymax=latency_max_log,
        )
        for x, y in points:
            canvas.circle(x, y, 4, fill=colors[arm])

    legends = [
        (arm, colors[arm]) for arm in ("adaptive", "fixed_k2", "fixed_k4", "fixed_k6")
    ]
    for panel in (0, 2, 3):
        for index, (label, color) in enumerate(legends):
            canvas.text(boxes[panel][0] + 280 + index * 180, boxes[panel][1] - 28, label, fill=color, small=True)
    for index, (label, color) in enumerate(pressure_colors.items()):
        canvas.text(boxes[1][0] + 430 + index * 220, boxes[1][1] - 28, label, fill=color, small=True)
    canvas.text(700, 1332, "Time since measured trace start (s)", small=True)

    for event in adaptive.get("resize_events") or []:
        if event.get("direction") not in {"grow", "shrink"}:
            continue
        for box in boxes[:3]:
            x = box[0] + float(event["elapsed_s"]) / xmax * (box[2] - box[0])
            canvas.line(
                [(x, box[1]), (x, box[3])],
                fill="#0072B2" if event["direction"] == "grow" else "#009E73",
                width=1,
            )

    output_dir = (args.output_dir or args.run_root / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "e13_kv_pressure_tier_throughput_latency_timeseries"
    canvas.save(stem.with_suffix(".png"), stem.with_suffix(".svg"))
    print(json.dumps({"status": "ok", "png": str(stem.with_suffix('.png'))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
