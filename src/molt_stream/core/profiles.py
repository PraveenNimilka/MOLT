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
        "description": "Maximum throughput with higher thermal ceiling (82°C limit)",
        "thermal_control_mode": "dual-gear",
        "thermal_target_c": 82.0,
        "thermal_cruise_max_c": 80.0,
        "thermal_pause_seconds": 0.05,
        "thermal_abort_c": 85.0,
    },
    "balanced": {
        "name": "BALANCED",
        "description": "High throughput with Dual-Gear thermal pacing (74°C limit)",
        "thermal_control_mode": "dual-gear",
        "thermal_target_c": 74.0,
        "thermal_cruise_max_c": 65.0,
        "thermal_pause_seconds": 0.22,
        "thermal_abort_c": 85.0,
    },
    "cool": {
        "name": "COOL",
        "description": "Conservative thermal target for quiet operation or laptops (68°C limit)",
        "thermal_control_mode": "dual-gear",
        "thermal_target_c": 68.0,
        "thermal_cruise_max_c": 60.0,
        "thermal_pause_seconds": 0.30,
        "thermal_abort_c": 82.0,
    },
    "energy": {
        "name": "ENERGY",
        "description": "Optimized duty cycle for energy efficiency per token (70°C limit)",
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
    )
