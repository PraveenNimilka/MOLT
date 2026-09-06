"""Attempt-aware training timing; rolled-back work must not vanish from rates."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class UpdateAccounting:
    """Session-local counters. Compute includes host overhead, excludes pacing.

    Call once per attempted accumulated update, after synchronizing device work.
    Checkpoint/recovery and validation remain in end-to-end timing, not here.
    """

    committed_compute_seconds: float = 0.0
    discarded_compute_seconds: float = 0.0
    committed_tokens: int = 0
    discarded_tokens: int = 0
    attempts: int = 0

    def record(self, *, elapsed_seconds: float, pause_seconds: float,
               tokens: int, committed: bool) -> None:
        if (not math.isfinite(elapsed_seconds) or not math.isfinite(pause_seconds)
                or not 0 <= pause_seconds <= elapsed_seconds
                or type(tokens) is not int or tokens < 0):
            raise ValueError("Invalid update timing or token count")
        compute = elapsed_seconds - pause_seconds
        self.attempts += 1
        if committed:
            self.committed_compute_seconds += compute
            self.committed_tokens += tokens
        else:
            self.discarded_compute_seconds += compute
            self.discarded_tokens += tokens

    def summary(self) -> dict[str, float | int | None]:
        total = self.committed_compute_seconds + self.discarded_compute_seconds
        return {
            "compute_accounting_version": 2,
            "update_attempts": self.attempts,
            "update_compute_seconds": total,
            "committed_update_compute_seconds": self.committed_compute_seconds,
            "discarded_update_compute_seconds": self.discarded_compute_seconds,
            "update_compute_tokens_per_second": self.committed_tokens / total if total else None,
            "committed_update_compute_tokens_per_second": (
                self.committed_tokens / self.committed_compute_seconds
                if self.committed_compute_seconds else None
            ),
        }
