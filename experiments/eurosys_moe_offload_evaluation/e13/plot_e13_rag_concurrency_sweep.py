#!/usr/bin/env python3
"""Export CSV and SVG for a verified E13 RAG B1--B8 sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = (
    ("fixed_k2", "Fixed-K2", "#0072B2"),
    ("fixed_k4", "Fixed-K4", "#E69F00"),
    ("fixed_k6", "Fixed-K6", "#009E73"),
    ("roundtrip", "Adaptive", "#CC79A7"),
)


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

    with (args.output_dir / "e13_rag_b1_b8_throughput.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["concurrency", "arm", "output_tok_s"])
        for batch in range(1, 9):
            for arm, display, _ in ARMS:
                writer.writerow([batch, display, matrix[f"b{batch}"][arm]["output_tok_s"]])

    width, height = 900, 540
    left, right, top, bottom = 92, 32, 58, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(
        float(matrix[f"b{batch}"][arm]["output_tok_s"])
        for batch in range(1, 9)
        for arm, _, _ in ARMS
    )
    y_max = max(20, int(math.ceil(maximum / 4.0) * 4))

    def x_coord(batch: int) -> float:
        return left + (batch - 1) / 7.0 * plot_width

    def y_coord(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.tick{font-size:13px}.label{font-size:15px}.title{font-size:19px;font-weight:600}.legend{font-size:14px}</style>',
        f'<text class="title" x="{width/2}" y="30" text-anchor="middle">KV-tier throughput across RAG-8192 concurrency</text>',
    ]
    for tick in range(0, y_max + 1, 4):
        y = y_coord(tick)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd"/>'
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
    for batch in range(1, 9):
        x = x_coord(batch)
        svg.append(
            f'<text class="tick" x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle">{batch}</text>'
        )
    svg.append(
        f'<text class="label" x="{left+plot_width/2:.2f}" y="{height-18}" text-anchor="middle">Concurrency</text>'
    )
    svg.append(
        f'<text class="label" transform="translate(24 {top+plot_height/2:.2f}) rotate(-90)" text-anchor="middle">Effective output throughput (tok/s)</text>'
    )
    for arm, display, color in ARMS:
        points = [
            (x_coord(batch), y_coord(float(matrix[f"b{batch}"][arm]["output_tok_s"])))
            for batch in range(1, 9)
        ]
        serialized = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        svg.append(
            f'<polyline points="{serialized}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
        for x, y in points:
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="white" stroke="{color}" stroke-width="2.6"/>'
            )
    legend_x, legend_y = left + 16, top + 20
    for index, (_, display, color) in enumerate(ARMS):
        column, row = index % 2, index // 2
        x = legend_x + column * 142
        y = legend_y + row * 27
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x+25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text class="legend" x="{x+34}" y="{y+5}">{display}</text>')
    svg.append("</svg>")
    (args.output_dir / "e13_rag_b1_b8_throughput.svg").write_text(
        "\n".join(svg) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
