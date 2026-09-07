from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "benchmarks" / "0.11.0a3-diagnostic.json"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "benchmark-diagnostic-0.11.0a3.svg"

COLORS = ("#2563eb", "#059669", "#d97706")


def _x(value: float, left: float, width: float, maximum: float) -> float:
    return left + width * value / maximum


def render(data: dict[str, Any]) -> str:
    width, height = 1200, 610
    families = data["results"]
    panels = (
        ("Elapsed time reduction", "time_reduction_percent", 90.0, True),
        ("Board energy reduction", "energy_reduction_percent", 70.0, True),
        ("Allocator peak reduction", "allocator_peak_reduction_percent", 10.0, False),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">MOLT 0.11.0a3 matched diagnostic reductions</title>',
        '<desc id="desc">Elapsed time, board energy, and PyTorch allocator peak reductions relative to matched Unsloth diagnostic arms for Qwen, Llama-family, and Gemma. Official promotion gates did not pass.</desc>',
        '<rect width="1200" height="610" fill="#ffffff"/>',
        '<text x="60" y="50" font-family="Segoe UI,Arial,sans-serif" font-size="26" font-weight="600" fill="#111827">MOLT 0.11.0a3 matched diagnostic screen</text>',
        '<text x="60" y="78" font-family="Segoe UI,Arial,sans-serif" font-size="15" fill="#4b5563">Reduction relative to tested Unsloth arms · positive favors MOLT · n=6 paired observations per family</text>',
    ]

    panel_width = 300.0
    plot_width = 220.0
    panel_lefts = (210.0, 555.0, 900.0)
    plot_top, plot_bottom = 145.0, 405.0
    rows = (195.0, 275.0, 355.0)

    for panel_index, (title, key, maximum, has_interval) in enumerate(panels):
        left = panel_lefts[panel_index]
        parts.extend(
            [
                f'<text x="{left:.1f}" y="118" font-family="Segoe UI,Arial,sans-serif" font-size="17" font-weight="600" fill="#111827">{escape(title)}</text>',
                f'<rect x="{left:.1f}" y="{plot_top:.1f}" width="{plot_width:.1f}" height="{plot_bottom - plot_top:.1f}" fill="#f9fafb" stroke="#d1d5db"/>',
            ]
        )
        for tick in range(0, 5):
            value = maximum * tick / 4
            xpos = _x(value, left, plot_width, maximum)
            parts.append(
                f'<line x1="{xpos:.1f}" y1="{plot_top:.1f}" x2="{xpos:.1f}" y2="{plot_bottom:.1f}" stroke="#e5e7eb"/>'
            )
            parts.append(
                f'<text x="{xpos:.1f}" y="430" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#4b5563">{value:.0f}</text>'
            )
        parts.append(
            f'<text x="{left + plot_width / 2:.1f}" y="456" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#374151">Reduction (%)</text>'
        )

        for index, result in enumerate(families):
            row = rows[index]
            metric = result[key]
            mean = metric["mean"] if has_interval else metric
            mean_x = _x(mean, left, plot_width, maximum)
            color = COLORS[index]
            if panel_index == 0:
                parts.append(
                    f'<text x="195" y="{row + 5:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="600" fill="#111827">{escape(result["family"])}</text>'
                )
            parts.append(
                f'<rect x="{left:.1f}" y="{row - 10:.1f}" width="{max(1.5, mean_x - left):.1f}" height="20" fill="{color}" opacity="0.20"/>'
            )
            if has_interval:
                low_x = _x(metric["ci95_low"], left, plot_width, maximum)
                high_x = _x(metric["ci95_high"], left, plot_width, maximum)
                parts.extend(
                    [
                        f'<line x1="{low_x:.1f}" y1="{row:.1f}" x2="{high_x:.1f}" y2="{row:.1f}" stroke="{color}" stroke-width="3"/>',
                        f'<line x1="{low_x:.1f}" y1="{row - 8:.1f}" x2="{low_x:.1f}" y2="{row + 8:.1f}" stroke="{color}" stroke-width="2"/>',
                        f'<line x1="{high_x:.1f}" y1="{row - 8:.1f}" x2="{high_x:.1f}" y2="{row + 8:.1f}" stroke="{color}" stroke-width="2"/>',
                    ]
                )
            parts.append(
                f'<circle cx="{mean_x:.1f}" cy="{row:.1f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
            )
            label = f"{mean:.2f}%"
            label_x = min(left + plot_width - 4, mean_x + 10)
            anchor = "end" if label_x == left + plot_width - 4 else "start"
            parts.append(
                f'<text x="{label_x:.1f}" y="{row - 17:.1f}" text-anchor="{anchor}" font-family="Segoe UI,Arial,sans-serif" font-size="12" font-weight="600" fill="#111827">{label}</text>'
            )

    parts.extend(
        [
            '<line x1="60" y1="492" x2="1140" y2="492" stroke="#d1d5db"/>',
            '<text x="60" y="522" font-family="Segoe UI,Arial,sans-serif" font-size="15" font-weight="600" fill="#991b1b">Diagnostic only — official promotion gate: NOT PASSED</text>',
            '<text x="60" y="550" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#4b5563">Time and energy whiskers: two-sided 95% paired t-intervals. Allocator panel: observed point reduction; no sampling interval.</text>',
            '<text x="60" y="575" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#4b5563">Clocks were not locked; one Gemma pair had a power-envelope mismatch. Soup, 7B/8B endurance, and independent reproduction remain open.</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the public MOLT benchmark summary SVG"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
