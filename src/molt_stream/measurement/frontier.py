from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QualityEnergyPoint:
    run_id: str
    seconds: float
    gpu_board_joules: float
    validation_perplexity: float


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
