"""Summarize GPU kernels separately from host waits; never sum both as wall time."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(trace: dict) -> dict:
    groups = {category: defaultdict(lambda: [0, 0.0]) for category in ("kernel", "cuda_runtime")}
    for event in trace.get("traceEvents", []):
        category = event.get("cat")
        if category not in groups or event.get("ph") != "X":
            continue
        row = groups[category][event["name"]]
        row[0] += 1
        row[1] += float(event.get("dur", 0))
    result = {}
    for category, values in groups.items():
        total = sum(row[1] for row in values.values())
        result[category] = {
            "summed_duration_us": total,
            "operations": [
                {"name": name, "calls": row[0], "duration_us": row[1],
                 "share_of_category": row[1] / total if total else 0.0}
                for name, row in sorted(values.items(), key=lambda item: item[1][1], reverse=True)
            ],
        }
    result["interpretation"] = (
        "Instrumented attribution only. Kernel sums can overlap across streams. "
        "Host synchronization includes device execution already counted as kernels; "
        "it is not independently removable overhead. Missing kernels are not zero work."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.trace.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
