from __future__ import annotations

from pathlib import Path

import torch

from molt_stream.core.specs import DataSpec


class MMapTokenBatcher:
    """File-size-independent virtual mapping with deterministic packed windows."""

    def __init__(self, spec: DataSpec, *, seed: int, device: torch.device | str):
        spec.validate()
        dtype = {"uint8": torch.uint8, "int32": torch.int32}[spec.storage_dtype]
        element_bytes = torch.empty((), dtype=dtype).element_size()
        size = Path(spec.path).stat().st_size
        if size % element_bytes:
            raise ValueError("token file is not aligned to storage dtype")
        count = size // element_bytes
        if count <= spec.context_length:
            raise ValueError("token file has no complete context window")
        self.spec = spec
        self.tokens = torch.from_file(spec.path, shared=False, size=count, dtype=dtype)
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.cursor = 0
        self.samples = 0

    @property
    def storage_backend(self) -> str:
        return "mmap"

    @property
    def mapped_file_bytes(self) -> int:
        return Path(self.spec.path).stat().st_size

    def batch(
        self, batch_size: int, *, context_length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        context = self.spec.context_length if context_length is None else context_length
        if context <= 0 or context > self.spec.context_length:
            raise ValueError(
                "runtime context_length must be positive and no larger than the mapped data spec"
            )
        high = len(self.tokens) - context - 1
        if self.spec.sequential:
            offsets = (self.cursor + torch.arange(batch_size) * context) % (high + 1)
            self.cursor = int((self.cursor + batch_size * context) % (high + 1))
        else:
            offsets = torch.randint(0, high + 1, (batch_size,), generator=self.generator)
        end = int(offsets[0]) + batch_size * context
        contiguous = (
            self.spec.packing == "contiguous"
            and self.spec.sequential
            and end < len(self.tokens)
        )
        if contiguous:
            start = int(offsets[0])
            windows = self.tokens[start : end + 1].unfold(0, context + 1, context)
        else:
            index = offsets[:, None] + torch.arange(context + 1)[None, :]
            windows = self.tokens[index]
        self.samples += batch_size
        compact = windows.to(self.device, non_blocking=self.device.type == "cuda").long()
        return compact[:, :-1], compact[:, 1:]

    def state_dict(self) -> dict[str, object]:
        return {
            "cursor": self.cursor,
            "samples": self.samples,
            "generator_state": self.generator.get_state(),
            "path": self.spec.path,
            "context_length": self.spec.context_length,
            "packing": self.spec.packing,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for field, expected in (
            ("path", self.spec.path),
            ("context_length", self.spec.context_length),
            ("packing", self.spec.packing),
        ):
            if state[field] != expected:
                raise ValueError(f"batcher checkpoint {field} mismatch")
        self.cursor = int(state["cursor"])
        self.samples = int(state["samples"])
        self.generator.set_state(state["generator_state"])  # type: ignore[arg-type]
