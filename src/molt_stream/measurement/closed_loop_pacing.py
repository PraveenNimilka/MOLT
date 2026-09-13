"""Experimental between-update pacing; never changes optimizer or kernel math."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ClosedLoopPacer:
    target_c: float = 70.0
    target_watts: float = 70.0
    thermal_duty: float = 0.8
    power_duty: float = 0.8
    previous_time: float | None = None
    previous_c: float | None = None
    slope: float = 0.0
    filtered_watts: float | None = None

    def __post_init__(self):
        if not (math.isfinite(self.target_c) and 0 < self.target_c < 80
                and math.isfinite(self.target_watts) and self.target_watts > 0):
            raise ValueError("invalid pacing targets")

    def delay(self, now: float, temperature_c: float, watts: float | None,
              compute_seconds: float, elapsed_seconds: float) -> float:
        if not all(math.isfinite(v) for v in (now, temperature_c, compute_seconds, elapsed_seconds)):
            raise ValueError("nonfinite pacing input")
        if temperature_c >= 84 or min(compute_seconds, elapsed_seconds) < 0:
            raise ValueError("unsafe temperature or timing")
        if self.previous_time is not None and now <= self.previous_time:
            raise ValueError("pacing timestamps must increase")
        dt = min(1.0, now - self.previous_time) if self.previous_time is not None else 0.1
        if self.previous_c is not None:
            rate = (temperature_c - self.previous_c) / dt
            alpha = 1 - math.exp(-dt / 2)
            self.slope += alpha * (rate - self.slope)
        predicted = temperature_c + 5 * max(0.0, self.slope)
        self.thermal_duty = min(1.0, max(0.05,
            self.thermal_duty + 0.012 * (self.target_c - predicted) * dt))
        if watts is not None:
            if not math.isfinite(watts) or watts < 0:
                raise ValueError("invalid power measurement")
            alpha = 1 - math.exp(-dt / 2)
            self.filtered_watts = watts if self.filtered_watts is None else (
                self.filtered_watts + alpha * (watts - self.filtered_watts))
            self.power_duty = min(1.0, max(0.05,
                self.power_duty + 0.002 * (self.target_watts - self.filtered_watts) * dt))
        self.previous_time, self.previous_c = now, temperature_c
        duty = min(self.thermal_duty, self.power_duty)
        # Charge all host work already elapsed; never add an unaccounted delay.
        return max(0.0, compute_seconds / duty - elapsed_seconds)
