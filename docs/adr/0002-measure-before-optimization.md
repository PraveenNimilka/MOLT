# ADR-0002: PyTorch eager/SDPA baseline before compilers or external kernels

- Status: Accepted for Milestone 0 proposal
- Date: 2026-09-01

## Context

The audited Windows machine has an NVIDIA driver but no Python, CUDA Toolkit, or
C/C++ build toolchain. External FlashAttention is primarily documented for Linux
and a compiled toolchain. Compilers also introduce startup and cache effects.

## Decision

BL-0001 uses official CUDA-enabled PyTorch wheels, AMP, and PyTorch scaled-dot-
product attention without `torch.compile` or external custom kernels. Record the
actual SDPA backend. Evaluate compile and kernel changes later as isolated
end-to-end experiments including installation assumptions, compilation, cache,
warm-up, correctness, and fresh-run time-to-quality.

## Consequences

The baseline is less aggressively optimized but more portable and attributable.
An optimization can be promoted only after it beats this strong framework-native
baseline under both warm-cache and fresh-run reporting.
