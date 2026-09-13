"""MOLT High-Level Training Profiles.

Defines user-facing policy profiles mapping to existing configuration
abstractions without modifying underlying controller mathematics:
- SPEED: Maximum throughput, minimal cooling pacing.
- BALANCED: High throughput with closed-loop Dual-Gear thermal protection.
- COOL: Conservative thermal target for warm ambient environments or quiet fans.
- ENERGY: Power-conscious duty cycle targeting optimized energy per token.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from molt_stream.core.specs import TrainingSpec

PROFILES: dict[str, dict[str, Any]] = {
    "speed": {
        "name": "SPEED",
        "description": "Less pacing: begins at 82°C; abort boundary 85°C",
        "thermal_control_mode": "dual-gear",
        "thermal_target_c": 82.0,
        "thermal_cruise_max_c": 80.0,
        "thermal_pause_seconds": 0.05,
        "thermal_abort_c": 85.0,
    },
    "micro": {
        "name": "MICRO",
        "description": "Experimental power-aware governor: bounded 1-100ms pacing; 84°C abort",
        "thermal_control_mode": "micro-guard",
        "thermal_target_c": 78.0,
        "thermal_cruise_max_c": 82.0,
        "thermal_microbatch_guard_c": 83.0,
        "thermal_pause_seconds": 0.04,
        "thermal_abort_c": 84.0,
        "thermal_power_target_watts": 48.0,
        "min_pause_ms": 1.0,
        "max_pause_ms": 12.0,
        "thermal_protective_pause_ms": 100.0,
        "thermal_protective_hysteresis_c": 2.0,
    },
    "balanced": {
        "name": "BALANCED",
        "description": "Balanced pacing: begins at 74°C; abort boundary 85°C",
        "thermal_control_mode": "dual-gear",
        "thermal_target_c": 74.0,
        "thermal_cruise_max_c": 65.0,
        "thermal_pause_seconds": 0.22,
        "thermal_abort_c": 85.0,
    },
    "cool": {
        "name": "COOL",
        "description": "Earlier pacing: begins at 68°C; abort boundary 82°C",
        "thermal_control_mode": "dual-gear",
        "thermal_target_c": 68.0,
        "thermal_cruise_max_c": 60.0,
        "thermal_pause_seconds": 0.30,
        "thermal_abort_c": 82.0,
    },
    "energy": {
        "name": "ENERGY",
        "description": "Energy experiment preset: pacing at 70°C; savings are workload-dependent",
        "thermal_control_mode": "dual-gear",
        "thermal_target_c": 70.0,
        "thermal_cruise_max_c": 62.0,
        "thermal_pause_seconds": 0.25,
        "thermal_abort_c": 85.0,
    },
}


def get_profile(name: str) -> dict[str, Any]:
    key = name.strip().lower()
    if key not in PROFILES:
        valid = ", ".join(k.upper() for k in PROFILES)
        raise ValueError(f"Unknown training profile '{name}'. Supported profiles: {valid}")
    return PROFILES[key]


def apply_profile(spec: TrainingSpec, profile_name: str) -> TrainingSpec:
    """Apply a named profile to a TrainingSpec."""
    prof = get_profile(profile_name)
    return replace(
        spec,
        thermal_control_mode=prof["thermal_control_mode"],
        thermal_target_c=prof["thermal_target_c"],
        thermal_cruise_max_c=prof["thermal_cruise_max_c"],
        thermal_pause_seconds=prof["thermal_pause_seconds"],
        thermal_abort_c=prof["thermal_abort_c"],
        thermal_power_target_watts=prof.get(
            "thermal_power_target_watts", spec.thermal_power_target_watts
        ),
        thermal_microbatch_guard_c=prof.get("thermal_microbatch_guard_c", spec.thermal_microbatch_guard_c),
        min_pause_ms=prof.get("min_pause_ms", spec.min_pause_ms),
        max_pause_ms=prof.get("max_pause_ms", spec.max_pause_ms),
        thermal_protective_pause_ms=prof.get(
            "thermal_protective_pause_ms", spec.thermal_protective_pause_ms
        ),
        thermal_protective_hysteresis_c=prof.get(
            "thermal_protective_hysteresis_c", spec.thermal_protective_hysteresis_c
        ),
    )
