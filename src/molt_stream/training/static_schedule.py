from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class LayerEvent:
    phase: str
    layer: int
    microbatch: int


@dataclass(frozen=True)
class AccumulationSchedule:
    """Deterministic transformer traversal for one optimizer update.

    This is a semantic schedule, not an execution claim.  The layer-major form
    is the contract for a future bounded NF4 decode cache: every parameter sees
    microbatches in the same order as the reference accumulation loop.
    """

    layers: int
    microbatches: int
    order: str
    events: tuple[LayerEvent, ...]

    @property
    def fingerprint(self) -> str:
        payload = {
            "layers": self.layers,
            "microbatches": self.microbatches,
            "order": self.order,
            "events": [event.__dict__ for event in self.events],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def build_accumulation_schedule(
    layers: int, microbatches: int, *, order: str
) -> AccumulationSchedule:
    if layers < 1 or microbatches < 1:
        raise ValueError("layers and microbatches must be positive")
    if order not in {"microbatch-major", "layer-major"}:
        raise ValueError("order must be microbatch-major or layer-major")

    events: list[LayerEvent] = []
    if order == "microbatch-major":
        for microbatch in range(microbatches):
            events.extend(LayerEvent("forward", layer, microbatch) for layer in range(layers))
            events.extend(
                LayerEvent("backward", layer, microbatch)
                for layer in range(layers - 1, -1, -1)
            )
    else:
        for layer in range(layers):
            events.extend(
                LayerEvent("forward", layer, microbatch)
                for microbatch in range(microbatches)
            )
        for layer in range(layers - 1, -1, -1):
            events.extend(
                LayerEvent("backward", layer, microbatch)
                for microbatch in range(microbatches)
            )
    return AccumulationSchedule(layers, microbatches, order, tuple(events))
