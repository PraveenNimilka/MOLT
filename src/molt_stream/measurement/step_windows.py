"""Step-window rates including evaluation and thermal pacing, not peak speed."""
from __future__ import annotations

import math


def step_window_rates(intervals: list[float], tokens_per_update: int) -> dict[str, float | None]:
    if tokens_per_update <= 0 or any(not math.isfinite(x) or x <= 0 for x in intervals):
        raise ValueError("Step intervals and tokens per update must be finite and positive")
    if len(intervals) < 8:
        return {"first_quarter_tokens_per_second": None,
                "last_quarter_tokens_per_second": None, "last_to_first_rate_ratio": None}
    count = len(intervals) // 4
    first = count * tokens_per_update / sum(intervals[:count])
    last = count * tokens_per_update / sum(intervals[-count:])
    return {"first_quarter_tokens_per_second": first,
            "last_quarter_tokens_per_second": last, "last_to_first_rate_ratio": last / first}
