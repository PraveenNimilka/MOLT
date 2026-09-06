"""Conservative paired replay screening; never promotes a research milestone."""
from __future__ import annotations

import math
from typing import Any


def compare_replay(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete/unmatched trials; disclose the host-memory tradeoff."""
    reasons: list[str] = []
    if baseline.get("state") != "completed" or candidate.get("state") != "completed":
        reasons.append("incomplete trial")
    if (baseline.get("spec") != candidate.get("spec")
            or baseline.get("versions") != candidate.get("versions")
            or baseline.get("harness_sha256") != candidate.get("harness_sha256")):
        reasons.append("workload/software mismatch")
    for key in ("windows", "epochs", "forward_guard", "start_temperature_c"):
        if baseline.get("probe", {}).get(key) != candidate.get("probe", {}).get(key):
            reasons.append(f"protocol mismatch: {key}")
    if baseline.get("probe", {}).get("cache") is not False or candidate.get("probe", {}).get("cache") is not True:
        reasons.append("expected uncached baseline and cached candidate")
    if not baseline.get("committed_tokens") or baseline.get("committed_tokens") != candidate.get("committed_tokens"):
        reasons.append("unequal committed tokens")
    for trial in (baseline, candidate):
        if len(trial.get("evaluations", [])) < 2:
            reasons.append("initial and final held-out evaluations required")
        if not trial.get("spec") or not trial.get("versions") or not trial.get("harness_sha256"):
            reasons.append("missing provenance")
    try:
        bn = float(baseline["evaluations"][-1]["nll"])
        cn = float(candidate["evaluations"][-1]["nll"])
        bt, ct = float(baseline["seconds"]), float(candidate["seconds"])
        bj = float(baseline["telemetry"]["gpu_board_energy_joules"])
        cj = float(candidate["telemetry"]["gpu_board_energy_joules"])
        if not all(math.isfinite(x) and x > 0 for x in (bn, cn, bt, ct, bj, cj)):
            raise ValueError("non-finite measurements")
        deltas = [abs(float(b["loss"]) - float(c["loss"]))
                  for b, c in zip(baseline["updates"], candidate["updates"], strict=True)]
        if not deltas or not all(math.isfinite(x) and x <= 1e-6 for x in deltas):
            reasons.append("training loss trajectory mismatch")
        if cn > bn * 1.01:
            reasons.append("quality regression exceeds 1%")
        if ct > bt * 0.99 or cj > bj * 0.99:
            reasons.append("less than 1% joint time/energy reduction")
        if candidate["peak_allocated_vram_bytes"] > baseline["peak_allocated_vram_bytes"]:
            reasons.append("allocated VRAM regression")
        bp = max(baseline["peak_gate_temperature_c"], baseline["telemetry"]["peak_gpu_temperature_c"])
        cp = max(candidate["peak_gate_temperature_c"], candidate["telemetry"]["peak_gpu_temperature_c"])
        if cp > min(72, bp):
            reasons.append("thermal regression")
        rss_delta = candidate["telemetry"]["peak_process_rss_bytes"] - baseline["telemetry"]["peak_process_rss_bytes"]
        if rss_delta > candidate["probe"]["cache_mib"] * 1024**2:
            reasons.append("RSS increase exceeds cache experiment budget")
        metrics = {"time_reduction_percent": 100 * (1 - ct / bt),
                   "board_energy_reduction_percent": 100 * (1 - cj / bj),
                   "final_nll_difference": cn - bn, "maximum_training_loss_difference": max(deltas),
                   "peak_rss_increase_bytes": rss_delta}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        reasons.append(f"invalid or missing measurements: {exc}")
        metrics = {}
    return {"decision": "extend-screen" if not reasons else "reject-or-inconclusive",
            "milestone_passed": False, "reasons": reasons, **metrics,
            "limitations": ["Repeated small subset; one-window held-out evaluation",
                            "Extra host memory is explicit, not a no-regression claim",
                            "No CPU thermal or independent reproduction evidence"]}


def main() -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    pairs = []
    for seed in (1337, 1338, 1339):
        paths = [root / f"replay-guarded-{arm}-8w4e-seed{seed}.json"
                 for arm in ("baseline", "cache")]
        baseline, candidate = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        pairs.append({"seed": seed, "artifacts": [str(path) for path in paths],
                      **compare_replay(baseline, candidate)})
    accepted = all(pair["decision"] == "extend-screen" for pair in pairs)
    result = {"decision": "extend-screen" if accepted else "reject-or-inconclusive",
              "milestone_passed": False, "pairs": pairs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
