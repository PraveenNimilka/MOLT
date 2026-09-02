import pytest

from molt_stream.training.curriculum import (
    CurriculumPhase,
    curriculum_geometry,
    interpolate_crossing,
    paired_improvement,
)


def test_curriculum_geometry_keeps_tokens_per_update_constant():
    phases = (
        CurriculumPhase(0.30, 128),
        CurriculumPhase(0.60, 256),
        CurriculumPhase(1.00, 512),
    )
    geometry = curriculum_geometry(
        max_steps=10,
        full_context=512,
        full_batch_size=24,
        phases=phases,
    )
    assert [(item.start_step, item.end_step, item.context_length, item.batch_size) for item in geometry] == [
        (0, 3, 128, 96),
        (3, 6, 256, 48),
        (6, 10, 512, 24),
    ]
    assert {item.context_length * item.batch_size for item in geometry} == {12_288}


def test_curriculum_rejects_non_divisible_geometry():
    with pytest.raises(ValueError, match="divide"):
        curriculum_geometry(
            max_steps=10,
            full_context=10,
            full_batch_size=3,
            phases=(CurriculumPhase(1.0, 4),),
        )


def test_interpolated_time_and_energy_crossing():
    crossing = interpolate_crossing(
        [
            {"nll": 4.0, "elapsed_seconds": 10.0, "energy_joules": 100.0},
            {"nll": 2.0, "elapsed_seconds": 30.0, "energy_joules": 260.0},
        ],
        target_nll=3.0,
    )
    assert crossing == {"seconds": 20.0, "energy_joules": 180.0}


def test_paired_gate_requires_time_energy_and_quality():
    result = paired_improvement(
        baseline={"final_nll": 3.0, "time_to_target_seconds": 100.0, "energy_to_target_joules": 1000.0},
        candidate={"final_nll": 3.02, "time_to_target_seconds": 75.0, "energy_to_target_joules": 740.0},
        quality_tolerance=0.01,
    )
    assert result["quality_within_tolerance"] is True
    assert result["time_improvement_percent"] == pytest.approx(25.0)
    assert result["energy_improvement_percent"] == pytest.approx(26.0)
    assert result["strong_gate"] is True


def test_quality_equivalent_target_allows_one_percent_regression():
    baseline_final_nll = 4.0
    assert baseline_final_nll * 1.01 == pytest.approx(4.04)
