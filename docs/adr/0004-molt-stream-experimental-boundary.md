# ADR 0004: Keep MOLT-Stream experimental and capability-gated

- Status: accepted
- Date: 2026-09-01

## Decision

`molt_stream` is isolated. Core has no outward dependencies; infrastructure
depends inward; streaming depends on core; training composes infrastructure; and
the CLI is outermost. An AST test enforces these ranks.

Liger, Triton and external QLoRA runtimes are optional capabilities. Missing
runtimes produce explicit errors. PyTorch fallbacks are labeled as fallbacks and
cannot satisfy a fused-kernel claim. Stable `ai_local` code never imports the
experimental streaming package.

## Rationale

This host lacks Triton, Liger, bitsandbytes, Transformers and PEFT. Treating
generic PyTorch operations as equivalent would invalidate benchmarks and novelty
claims. Isolation allows falsification without destabilizing the baseline.
