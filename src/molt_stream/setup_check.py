"""Small explicit installation checks, not a training performance benchmark."""
from __future__ import annotations

import argparse
import importlib
import json

import torch


def check_runtime(*, compile_enabled: bool = False) -> dict[str, object]:
    for name in ("accelerate", "bitsandbytes", "transformers", "peft"):
        importlib.import_module(name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and CUDA-enabled PyTorch installation.")

    def loss(x: torch.Tensor) -> torch.Tensor:
        return (x.sin() * x).sum()

    x = torch.linspace(-1, 1, 1024, device="cuda", requires_grad=True)
    expected = torch.autograd.grad(loss(x), x)[0]
    torch.cuda.synchronize()
    result: dict[str, object] = {"cuda_backward": "passed", "qlora_imports": "passed", "compile_backward": "not_requested"}
    if compile_enabled:
        compiled = torch.compile(loss, mode="max-autotune-no-cudagraphs", fullgraph=True)
        actual = torch.autograd.grad(compiled(x), x)[0]
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        result["compile_backward"] = "passed"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", dest="compile_enabled")
    args = parser.parse_args(argv)
    try:
        result = check_runtime(compile_enabled=args.compile_enabled)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps({"status": "passed", **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
