# 1B workload saturation experiment

## Question and interpretation rule

Can better workload geometry raise sustained token throughput for the dense 1B
model on the audited RTX 4060 Laptop while retaining at least 1 GiB estimated
VRAM headroom? A raw-throughput winner is only an engineering recommendation;
it is not a learning-efficiency result until an equal-quality control passes.

## Observed result

The eager PyTorch/SDPA sweep increased context length at micro-batch one. The
short benchmark measured 58.60 tokens/s at context 8, 963.40 at 128, 1,456.16 at
256, 1,901.59 at 512, 2,119.16 at 1,024, and 2,118.64 at 2,048. Throughput
saturated around context 1,024. At that point PyTorch peak allocation was 4.107
GB and the conservative estimated device headroom was 3.202 GB. Data enqueue
accounted for 0.041% of measured time.

Raw artifacts:

- `artifacts/benchmarks/saturation-20260901-020315.json`
- `artifacts/benchmarks/saturation-20260901-020349.json`

This is a short single-device benchmark without repetition-derived confidence
intervals. It demonstrates underutilization of tiny shapes, not data-center
equivalence.

## Equal-token falsification

EXP-1010 trained context 128 for 128 updates: the same 16,384 tokens used by
EXP-1005. It completed in 19.143 seconds (855.89 end-to-end tokens/s) and sampled
1,319.45 J of GPU board energy, versus 256.726 seconds and 17,219.17 J for
EXP-1005. However, validation ended at 5.03686 bits/byte rather than 2.92159 and
never reached the target. The model also has 245,760 additional position-
embedding parameters, a 0.024% difference, so the CLI correctly warns that this
is not a perfectly controlled parameter-count comparison.

Decision: retain memory mapping, vectorized byte transfer, phase measurements,
and safe shape search as platform capabilities. Reject the claim that shape
saturation alone improves equal-token learning efficiency. Next test schedules
that preserve optimizer-update count or explicitly adjust learning rate using a
small real-language baseline before spending energy on another 1B run.

## Compiler probe

An isolated `torch.compile(..., mode="reduce-overhead")` CUDA operation failed in
the pinned environment with `torch._inductor.exc.TritonMissing`. No package was
installed and eager mode remains explicit. Official PyTorch documentation says
the mode uses CUDA graphs to reduce launch overhead, but may increase memory and
does not support every model. A future compiler experiment therefore needs its
own dependency pin, correctness test, compile-time accounting, and eager
control; it is not part of the present result.
