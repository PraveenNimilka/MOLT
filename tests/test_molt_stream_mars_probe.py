import json

import pytest

from molt_stream.training.mars_probe import summarize_trials


@pytest.mark.parametrize("candidate_joules, decision", [
    (90.0, "revise"), (110.0, "reject-or-inconclusive"),
])
def test_cheap_probe_never_promotes_and_rejects_energy_regression(tmp_path, candidate_joules, decision):
    records = []
    for seed in (1337, 2027, 3407):
        for batch in (1, 2):
            path = tmp_path / f"{seed}-{batch}.json"
            path.write_text(json.dumps({
                "state": "completed", "evaluations": [{"nll": 2.0}],
                "telemetry": {"gpu_board_energy_joules": 100.0 if batch == 1 else candidate_joules},
            }), encoding="utf-8")
            records.append({"seed": seed, "batch_size": batch, "returncode": 0,
                            "summary": str(path), "process_seconds": 10.0 if batch == 1 else 8.0})
    (tmp_path / "manifest.json").write_text(json.dumps({"trials": records}), encoding="utf-8")
    report = summarize_trials(tmp_path)
    assert report["decision"] == decision
    assert report["milestone_passed"] is False
    assert (tmp_path / "screen.json").exists()
