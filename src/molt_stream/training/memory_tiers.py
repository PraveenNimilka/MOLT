"""Opt-in resident memory planning; measured fit still overrides size heuristics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidentMemoryTier:
    name: str
    activation_checkpointing: bool
    reserve_bytes: int
    qualified: bool = False


def plan_resident_tier(parameter_count: int, available_bytes: int) -> ResidentMemoryTier:
    """Use the original dense parameter count, not packed NF4 storage elements."""
    if type(parameter_count) is not int or type(available_bytes) is not int or parameter_count <= 0 or available_bytes <= 0:
        raise ValueError("positive model size and measured available memory required")
    reserve = max(512 * 2**20, available_bytes // 10)
    if available_bytes <= reserve:
        raise ValueError("available memory cannot retain the safety reserve")
    if parameter_count <= 3_000_000_000:
        return ResidentMemoryTier("resident-small", False, reserve)
    if parameter_count >= 7_000_000_000:
        return ResidentMemoryTier("resident-large-recompute", True, reserve)
    return ResidentMemoryTier("resident-middle-fit-test", False, reserve)
