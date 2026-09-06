from molt_stream.experiments.replay_screen import compare_replay


def test_incomplete_pair_cannot_pass_screen():
    result = compare_replay({"state": "thermal_stop"}, {"state": "completed"})
    assert result["decision"] == "reject-or-inconclusive"
    assert not result["milestone_passed"]
    assert "incomplete trial" in result["reasons"]


def test_missing_measurements_cannot_count_as_zero_energy():
    result = compare_replay({}, {})
    assert any("missing measurements" in reason for reason in result["reasons"])
