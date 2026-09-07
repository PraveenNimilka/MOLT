from __future__ import annotations

from collections import Counter

import pytest

from molt_stream.training.static_schedule import build_accumulation_schedule


@pytest.mark.parametrize("order", ["microbatch-major", "layer-major"])
def test_schedule_visits_every_layer_microbatch_pair_once_per_phase(order: str) -> None:
    schedule = build_accumulation_schedule(3, 4, order=order)
    observed = Counter((event.phase, event.layer, event.microbatch) for event in schedule.events)
    expected = Counter(
        (phase, layer, microbatch)
        for phase in ("forward", "backward")
        for layer in range(3)
        for microbatch in range(4)
    )
    assert observed == expected


def test_layer_major_preserves_microbatch_gradient_order_per_parameter() -> None:
    schedule = build_accumulation_schedule(3, 4, order="layer-major")
    for layer in range(3):
        assert [
            event.microbatch
            for event in schedule.events
            if event.phase == "backward" and event.layer == layer
        ] == [0, 1, 2, 3]


def test_schedule_fingerprint_is_deterministic_and_topology_sensitive() -> None:
    reference = build_accumulation_schedule(28, 4, order="microbatch-major")
    assert reference.fingerprint == build_accumulation_schedule(
        28, 4, order="microbatch-major"
    ).fingerprint
    assert reference.fingerprint != build_accumulation_schedule(
        28, 4, order="layer-major"
    ).fingerprint


@pytest.mark.parametrize("layers,microbatches", [(0, 1), (1, 0)])
def test_invalid_schedule_is_rejected(layers: int, microbatches: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_accumulation_schedule(layers, microbatches, order="layer-major")
