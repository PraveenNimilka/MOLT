from __future__ import annotations

import pytest

from molt_stream.measurement.update_regions import (
    attribute_update_energy,
    integrate_power_window,
)


def test_power_window_clips_and_interpolates_boundaries() -> None:
    points = [
        {"monotonic_seconds": 0.0, "gpu_power_watts": 10.0},
        {"monotonic_seconds": 2.0, "gpu_power_watts": 30.0},
        {"monotonic_seconds": 4.0, "gpu_power_watts": 10.0},
    ]
    observed = integrate_power_window(points, 1.0, 3.0)
    assert observed.joules == pytest.approx(50.0)
    assert observed.covered_seconds == pytest.approx(2.0)


def test_energy_partition_sums_to_measured_window_without_sync_double_count() -> None:
    update = {
        "start": 0.0,
        "end": 1.0,
        "regions": {
            "dataset_transfer": {"host_seconds": 0.10, "cuda_seconds": 0.05},
            "graph_input_staging": {"host_seconds": 0.02, "cuda_seconds": 0.02},
            "transformer_graph_replay": {"host_seconds": 0.0, "cuda_seconds": 0.50},
            "frozen_vocabulary_loss": {"host_seconds": 0.0, "cuda_seconds": 0.10},
            "optimizer_update": {"host_seconds": 0.03, "cuda_seconds": 0.08},
            "cuda_synchronization": {"host_seconds": 0.60, "cuda_seconds": None},
            "telemetry": {"host_seconds": 0.02, "cuda_seconds": None},
            "thermal_pacing": {"host_seconds": 0.10, "cuda_seconds": None},
            "python_dispatch": {"host_seconds": 0.13, "cuda_seconds": None},
        },
    }
    points = [
        {"monotonic_seconds": 0.0, "gpu_power_watts": 50.0},
        {"monotonic_seconds": 1.0, "gpu_power_watts": 50.0},
    ]
    observed = attribute_update_energy(update, points, idle_board_watts=10.0)
    estimates = [
        value["estimated_board_joules"]
        for value in observed["regions"].values()
    ]
    assert sum(estimates) == pytest.approx(50.0)
    assert observed["regions"]["cuda_synchronization"]["estimated_board_joules"] == 0.0
    assert observed["regions"]["cuda_synchronization"]["non_additive_overlap"] is True
