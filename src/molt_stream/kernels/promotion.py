from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelMeasurement:
    median_ms: float
    peak_allocated_bytes: int

    def __post_init__(self) -> None:
        if self.median_ms <= 0:
            raise ValueError("kernel latency must be positive")
        if self.peak_allocated_bytes < 0:
            raise ValueError("kernel peak allocation cannot be negative")


@dataclass(frozen=True)
class KernelPromotion:
    promoted: bool
    speedup_percent: float
    memory_change_percent: float
    reason: str


def evaluate_kernel_candidate(
    reference: KernelMeasurement,
    candidate: KernelMeasurement,
    *,
    minimum_speedup_percent: float = 5.0,
    maximum_memory_growth_percent: float = 0.0,
    parity_passed: bool,
) -> KernelPromotion:
    """Apply MOLT's fail-closed component promotion policy.

    A candidate must pass numerical parity, exceed the configured latency
    margin, and avoid material allocation growth.  The margin prevents timer
    noise and driver jitter from promoting a kernel with no practical benefit.
    """

    if minimum_speedup_percent < 0 or maximum_memory_growth_percent < 0:
        raise ValueError("promotion margins cannot be negative")
    speedup = (reference.median_ms - candidate.median_ms) / reference.median_ms * 100.0
    if reference.peak_allocated_bytes == 0:
        memory_change = 0.0 if candidate.peak_allocated_bytes == 0 else float("inf")
    else:
        memory_change = (
            candidate.peak_allocated_bytes - reference.peak_allocated_bytes
        ) / reference.peak_allocated_bytes * 100.0
    if not parity_passed:
        reason = "numerical parity failed"
    elif speedup < minimum_speedup_percent:
        reason = (
            f"speedup {speedup:.2f}% is below the required "
            f"{minimum_speedup_percent:.2f}%"
        )
    elif memory_change > maximum_memory_growth_percent:
        reason = (
            f"memory growth {memory_change:.2f}% exceeds the allowed "
            f"{maximum_memory_growth_percent:.2f}%"
        )
    else:
        reason = "parity, latency, and memory gates passed"
    return KernelPromotion(
        promoted=reason == "parity, latency, and memory gates passed",
        speedup_percent=speedup,
        memory_change_percent=memory_change,
        reason=reason,
    )
