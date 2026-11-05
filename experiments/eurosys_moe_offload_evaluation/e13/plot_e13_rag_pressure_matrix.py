#!/usr/bin/env python3
"""Plot verified E13 RAG pressure throughput curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = (
    ("fixed_k2", "Fixed-K2", "#0072B2", "o"),
    ("fixed_k4", "Fixed-K4", "#E69F00", "s"),
    ("fixed_k6", "Fixed-K6", "#009E73", "^"),
    ("roundtrip", "Adaptive", "#CC79A7", "D"),
)
PRESSURES = (("low", 1), ("medium", 5), ("high", 6))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    verified = json.loads(args.verification.read_text(encoding="utf-8"))
    if verified.get("status") != "ok":
        raise RuntimeError("refusing to plot an unverified result")
    matrix = verified["matrix"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "e13_rag_pressure_throughput.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pressure", "concurrency", "arm", "output_tok_s"])
        for pressure, batch in PRESSURES:
            for arm, display, _, _ in ARMS:
                writer.writerow(
                    [pressure, batch, display, matrix[pressure][arm]["output_tok_s"]]
                )

    width, height = 900, 540
    left, right, top, bottom = 92, 32, 58, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = 1, 6
    maximum = max(
        float(matrix[pressure][arm]["output_tok_s"])
        for pressure, _ in PRESSURES
        for arm, _, _, _ in ARMS
    )
    y_max = max(20, int(math.ceil(maximum / 4.0) * 4))

    def x_coord(batch: int) -> float:
        return left + (batch - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.tick{font-size:13px}.label{font-size:15px}.title{font-size:19px;font-weight:600}.legend{font-size:14px}</style>',
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">KV-tier performance crossover under increasing pressure</text>',
    ]
    for tick in range(0, y_max + 1, 4):
        y = y_coord(tick)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="tick" x="{left-12}" y="{y+5:.2f}" text-anchor="end">{tick}</text>'
        )
    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333" stroke-width="1.3"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333" stroke-width="1.3"/>',
        ]
    )
    pressure_labels = {"low": "Low", "medium": "Medium", "high": "High"}
    for pressure, batch in PRESSURES:
        x = x_coord(batch)
        svg.append(
            f'<line x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" y2="{height-bottom+6}" stroke="#333"/>'
        )
        svg.append(
            f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle">B{batch}</text>'
        )
        svg.append(
            f'<text class="tick" x="{x:.2f}" y="{height-bottom+40}" text-anchor="middle">{pressure_labels[pressure]}</text>'
        )
    svg.append(
        f'<text class="label" x="{left+plot_width/2:.2f}" y="{height-16}" text-anchor="middle">Concurrency (RAG-8192 → 16)</text>'
    )
    svg.append(
        f'<text class="label" transform="translate(24 {top+plot_height/2:.2f}) rotate(-90)" text-anchor="middle">Effective output throughput (tok/s)</text>'
    )

    for arm, display, color, _ in ARMS:
        points = [
            (x_coord(batch), y_coord(float(matrix[pressure][arm]["output_tok_s"])))
            for pressure, batch in PRESSURES
        ]
        serialized = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        svg.append(
            f'<polyline points="{serialized}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
        for x, y in points:
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="white" stroke="{color}" stroke-width="3"/>'
            )

    legend_x, legend_y = left + 18, top + 20
    for index, (_, display, color, _) in enumerate(ARMS):
        column, row = index % 2, index // 2
        x = legend_x + column * 142
        y = legend_y + row * 28
        svg.append(
            f'<line x1="{x}" y1="{y}" x2="{x+25}" y2="{y}" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<circle cx="{x+12.5}" cy="{y}" r="4" fill="white" stroke="{color}" stroke-width="2"/>'
        )
        svg.append(
            f'<text class="legend" x="{x+34}" y="{y+5}">{display}</text>'
        )
    svg.append("</svg>")
    (args.output_dir / "e13_rag_pressure_throughput.svg").write_text(
        "\n".join(svg) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
