"""Validated interpolation of measured time and energy to a held-out NLL target."""
from __future__ import annotations

import math
from typing import Any


def interpolate_nll_crossing(evaluations: list[dict[str, Any]],
                             target_nll: float) -> dict[str, float] | None:
    if not math.isfinite(target_nll):
        raise ValueError("target_nll must be finite")
    previous_time = previous_energy = previous_step = -math.inf
    for index, current in enumerate(evaluations):
        values = {
            "nll": float(current["nll"]), "seconds": float(current["end_to_end_seconds"]),
            "joules": float(current["board_energy_joules"]), "step": float(current["step"]),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("evaluation crossing inputs must be finite")
        if (values["seconds"] < previous_time or values["joules"] < previous_energy
                or values["step"] < previous_step):
            raise ValueError("evaluation time, energy, and step must be nondecreasing")
        if values["nll"] <= target_nll:
            if index == 0:
                return {key: values[key] for key in ("seconds", "joules", "step")}
            previous = evaluations[index - 1]
            high, low = float(previous["nll"]), values["nll"]
            fraction = 1.0 if high <= low else (high - target_nll) / (high - low)
            fraction = min(1.0, max(0.0, fraction))
            return {
                key: float(previous[source]) + fraction * (values[key] - float(previous[source]))
                for key, source in (("seconds", "end_to_end_seconds"),
                                    ("joules", "board_energy_joules"), ("step", "step"))
            }
        previous_time, previous_energy, previous_step = (
            values["seconds"], values["joules"], values["step"]
        )
    return None
