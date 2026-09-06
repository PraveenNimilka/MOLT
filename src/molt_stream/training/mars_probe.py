"""Cheap, isolated geometry falsification; not a MARS or Unsloth speed claim.

Run with ``python -m molt_stream.training.mars_probe --help``. Each training
trial gets a fresh process. The manifest retains failures and resolved specs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from molt_stream.core.specs import load_spec


def summarize_trials(root: Path) -> dict[str, object]:
    """Persist all paired outcomes; a cheap probe can reject, never promote."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    groups: dict[int, dict[int, tuple[dict, dict]]] = {}
    for record in manifest["trials"]:
        if record["returncode"] or record["summary"] is None:
            continue
        summary = json.loads(Path(record["summary"]).read_text(encoding="utf-8"))
        groups.setdefault(record["seed"], {})[record["batch_size"]] = (record, summary)
    pairs = []
    for seed, trials in sorted(groups.items()):
        if set(trials) != {1, 2}:
            continue
        baseline, candidate = trials[1], trials[2]
        b, c = baseline[1], candidate[1]
        bj = b["telemetry"]["gpu_board_energy_joules"]
        cj = c["telemetry"]["gpu_board_energy_joules"]
        nll_b, nll_c = b["evaluations"][-1]["nll"], c["evaluations"][-1]["nll"]
        measured = all(isinstance(x, (float, int)) and math.isfinite(x) and x > 0
                       for x in (bj, cj, nll_b, nll_c))
        acceptable = (measured and b["state"] == c["state"] == "completed"
                      and candidate[0]["process_seconds"] < baseline[0]["process_seconds"]
                      and cj <= bj and nll_c <= nll_b * 1.01)
        pairs.append({
            "seed": seed, "passes_cheap_screen": bool(acceptable),
            "baseline_process_seconds": baseline[0]["process_seconds"],
            "candidate_process_seconds": candidate[0]["process_seconds"],
            "baseline_board_joules": bj, "candidate_board_joules": cj,
            "baseline_final_nll": nll_b, "candidate_final_nll": nll_c,
        })
    report = {
        "decision": ("revise" if len(pairs) == 3 and
                     all(p["passes_cheap_screen"] for p in pairs) else "reject-or-inconclusive"),
        "milestone_passed": False, "pairs": pairs,
        "limitations": ["Short geometry screen, not time-to-target quality evidence",
                        "Board sampling excludes Python startup and process teardown",
                        "Background contention and initial temperatures not controlled",
                        "No CPU temperature sensor; no Unsloth comparison"],
    }
    (root / "screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    spec = load_spec(args.config)
    if not spec.data.validation_path:
        parser.error("A separate validation dataset is required")
    spec.validate()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for index, seed in enumerate((1337, 2027, 3407)):
        # Alternate order to expose (not eliminate) warm-up/thermal bias.
        for batch in ((1, 2) if index % 2 == 0 else (2, 1)):
            name = f"seed-{seed}-microbatch-{batch}"
            trial = root / name
            trial.mkdir()
            candidate = replace(
                spec, seed=seed, batch_size=batch, gradient_accumulation=4 // batch,
                max_steps=args.steps, evaluation_interval=max(1, args.steps // 3),
                artifacts_dir=str(trial / "runs"),
            )
            candidate.validate()
            config = trial / "config.json"
            config.write_text(json.dumps(candidate.to_dict(), indent=2), encoding="utf-8")
            started = time.perf_counter()
            with (trial / "stdout.txt").open("w", encoding="utf-8") as stdout, \
                 (trial / "stderr.txt").open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    [sys.executable, "-m", "molt_stream.cli", "--json", "train",
                     "--config", str(config)], stdout=stdout, stderr=stderr,
                    check=False,
                )
            summaries = list((trial / "runs").glob("*/metrics.summary.json"))
            record: dict[str, object] = {
                "seed": seed, "batch_size": batch, "effective_batch": 4,
                "returncode": result.returncode,
                "process_seconds": time.perf_counter() - started,
                "config": str(config),
                "summary": str(summaries[0]) if len(summaries) == 1 else None,
            }
            records.append(record)
            temporary = root / "manifest.tmp"
            temporary.write_text(json.dumps({
                "purpose": "geometry kill test, not milestone acceptance",
                "trials": records,
            }, indent=2), encoding="utf-8")
            temporary.replace(root / "manifest.json")
            print(json.dumps(record), flush=True)
            if result.returncode:
                return result.returncode
            summary = json.loads(summaries[0].read_text(encoding="utf-8")) if len(summaries) == 1 else {}
            if summary.get("state") != "completed":
                print("Probe stopped: trial incomplete (including thermal abort).", flush=True)
                return 1
    print(json.dumps(summarize_trials(root)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
