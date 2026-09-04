from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class QualityEnergyPoint:
    run_id: str
    seconds: float
    gpu_board_joules: float
    validation_perplexity: float


def build_frontier(runs: list[str]) -> dict[str, object]:
    points = []
    for run in runs:
        metrics = json.loads((Path(run) / "metrics.summary.json").read_text("utf-8"))
        if metrics.get("state") != "completed":
            raise ValueError(f"Frontier requires completed runs: {run}")
        energy = metrics.get("telemetry", {}).get("gpu_board_energy_joules")
        evaluations = metrics.get("evaluations", [])
        if energy is None or not evaluations:
            raise ValueError(f"Frontier requires measured board energy and validation: {run}")
        quality = float(evaluations[-1]["perplexity"])
        seconds = float(metrics["seconds"])
        if not all(math.isfinite(x) and x >= 0 for x in (seconds, float(energy), quality)):
            raise ValueError(f"Non-finite or negative frontier metrics: {run}")
        points.append(QualityEnergyPoint(str(Path(run).resolve()), seconds, float(energy), quality))
    return frontier_dict(points)


def pareto_frontier(points: list[QualityEnergyPoint]) -> list[QualityEnergyPoint]:
    """Return non-dominated points; time, energy and perplexity are minimized."""
    return [
        point
        for point in points
        if not any(
            other != point
            and other.seconds <= point.seconds
            and other.gpu_board_joules <= point.gpu_board_joules
            and other.validation_perplexity <= point.validation_perplexity
            and (
                other.seconds < point.seconds
                or other.gpu_board_joules < point.gpu_board_joules
                or other.validation_perplexity < point.validation_perplexity
            )
            for other in points
        )
    ]


def frontier_dict(points: list[QualityEnergyPoint]) -> dict[str, object]:
    frontier = pareto_frontier(points)
    return {
        "objective": "minimize end-to-end seconds, GPU board joules, and validation perplexity",
        "points": [asdict(point) for point in points],
        "pareto_run_ids": [point.run_id for point in frontier],
    }
