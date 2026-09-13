"""Summarize isolated screens, retaining paired seeds and failed gates."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import pairwise
from pathlib import Path


def window_energy(points, start, end):
    total = 0.0
    coverage = 0.0
    for left, right in pairwise(points):
        t0, t1 = left["monotonic_seconds"], right["monotonic_seconds"]
        lo, hi = max(start, t0), min(end, t1)
        p0, p1 = left.get("gpu_power_watts"), right.get("gpu_power_watts")
        if hi <= lo or t1 <= t0 or p0 is None or p1 is None:
            continue
        a = p0 + (p1 - p0) * (lo - t0) / (t1 - t0)
        b = p0 + (p1 - p0) * (hi - t0) / (t1 - t0)
        total += (a + b) * 0.5 * (hi - lo)
        coverage += hi - lo
    return {"joules": total if coverage else None, "covered_seconds": coverage,
            "window_seconds": end - start}


def summarize(root):
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["status"] != "completed":
            rows.append({"run": path.parent.name, "status": row["status"]})
            continue
        points = json.loads((path.parent / "telemetry.json").read_text(encoding="utf-8"))
        rows.append({"run": path.parent.name, "arm": row["arm"], "seed": row["seed"],
            "compute_tok_s": row["compute_tok_s"], "training_tok_s": row["training_tok_s"],
            "end_to_end_tok_s": row["end_to_end_tok_s"],
            "allocated_gib": row["peak_allocated_bytes"] / 2**30,
            "reserved_gib": row["peak_reserved_bytes"] / 2**30,
            "whole_gpu_gib": row["telemetry"]["peak_gpu_used_bytes"] / 2**30,
            "board_joules": row["telemetry"]["gpu_board_energy_joules"],
            "peak_c": row["telemetry"]["peak_gpu_temperature_c"],
            "initial_nll": row["initial_validation_nll"], "final_nll": row["final_validation_nll"],
            "adapter_sha256": row["initial_adapter_sha256"],
            "phase_seconds": {k: v["seconds"] for k, v in row["phases"].items()},
            "phase_energy": {k: window_energy(points, v["start"], v["end"])
                             for k, v in row["phases"].items()}})
    metrics = ("compute_tok_s", "training_tok_s", "end_to_end_tok_s", "allocated_gib",
               "reserved_gib", "whole_gpu_gib", "board_joules", "peak_c", "final_nll")
    aggregates = {}
    for arm in ("hf", "reference", "qualified", "unsloth"):
        group = [row for row in rows if row.get("arm") == arm]
        if group:
            aggregates[arm] = {"runs": len(group), **{key: statistics.mean(row[key] for row in group)
                                                    for key in metrics}}
    comparisons = {}
    # Each seed is one independent unit; reverse-order repeats are averaged
    # within seed, never counted as extra independent seeds.
    for other in ("hf", "reference", "unsloth"):
        paired = []
        for seed in sorted({row.get("seed") for row in rows if "seed" in row}):
            left = [row for row in rows if row.get("arm") == "qualified" and row["seed"] == seed]
            right = [row for row in rows if row.get("arm") == other and row["seed"] == seed]
            if len(left) != 2 or len(right) != 2:
                continue
            if len({row["adapter_sha256"] for row in left + right}) != 1:
                continue
            paired.append({key: statistics.mean(row[key] for row in left) /
                               statistics.mean(row[key] for row in right)
                           for key in metrics})
        if len(paired) >= 2:
            # Two-sided t critical values for 1..5 degrees of freedom.
            critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(paired)-1)
            entries = {}
            for key in metrics:
                values = [math.log(row[key]) for row in paired]
                mean = statistics.mean(values)
                margin = critical * statistics.stdev(values) / math.sqrt(len(values)) if critical else None
                entries[key] = {"qualified_over_other_geomean": math.exp(mean),
                    "paired_seed_95pct_t_interval": [math.exp(mean-margin), math.exp(mean+margin)] if margin is not None else None}
            comparisons[other] = {"independent_seeds": len(paired), "ratios": entries}
    return {"rows": rows, "means": aggregates, "paired_seed_comparisons": comparisons, "official_promotion": False,
            "reason": "screening only; no five-seed 1M-token controlled-clock acceptance"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["means"], indent=2))
