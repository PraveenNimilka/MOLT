# NF4-LoRA decoder scheduling screen

Date: 2026-09-06. Status: correctness passed, performance hypothesis rejected.
The candidate remains isolated under `molt_stream.methods` and is not reachable
from production configuration.

## Hypothesis

A manually scheduled autograd function might reduce PEFT graph overhead by
computing the frozen NF4 input gradient and LoRA A/B gradients together. The
independent variable was only the backward schedule. NF4 data, double-quantized
scales, LoRA rank and scaling, precision, tensors and upstream gradient were
identical.

The screen required a positive median latency improvement on representative
Qwen2.5-1.5B projection shapes with no gradient or memory regression. A marginal
win on only one shape was insufficient because every decoder layer contains the
other projections.

## Prior-art correction

The original premise that MOLT repeatedly performs a separate full-weight
dequantization before every forward GEMM is incorrect for the installed
bitsandbytes 0.48 path. `matmul_4bit` already dispatches a packed `gemm_4bit`
forward. Its training backward dequantizes the frozen matrix to calculate the
input gradient. QLoRA's NF4 and double quantization are established work, and
Unsloth already implements manually scheduled LoRA forward/backward paths.

Consequently, this experiment is an engineering ablation—not a new algorithm.
The remaining high-leverage proposal is a true packed-NF4 tensor-core backward
kernel that forms `dX` without publishing a dense dequantized matrix.

## Correctness

The clean-room function implements:

```text
Y  = NF4_GEMM(X, Wq) + s (X A^T) B^T
dX = dY dequant(Wq) + s (dY B) A
dA = s (dY B)^T X
dB = s dY^T (X A^T)
```

CUDA tests require exact forward equality and BF16/FP32 gradient agreement with
the standard bitsandbytes plus PEFT dataflow. CPU use is explicitly rejected.

## Valid randomized component results

All cases used 512 tokens, rank 8, NF4 block size 64, compressed statistics and
30 interleaved samples for the two MLP shapes (20 for Q projection).

| Projection `(in -> out)` | Reference median | Candidate median | Change | Peak allocation change |
|---|---:|---:|---:|---:|
| `1536 -> 1536` | 0.8340 ms | 0.9584 ms | 14.9% slower | 31.30 -> 29.37 MB |
| `1536 -> 8960` | 2.7110 ms | 2.7520 ms | 1.5% slower | 75.54 -> 73.76 MB |
| `8960 -> 1536` | 2.7546 ms | 2.6941 ms | 2.2% faster | 83.01 -> 73.70 MB |

The candidate reduced temporary memory modestly and improved only the down
projection. It lost on the square and up projections, so it cannot plausibly
close MOLT's measured 10.1% compute gap to the local Unsloth diagnostic. No
integrated run was launched. Two initially concurrent MLP artifacts are marked
invalid and excluded; the `*-valid.json` files contain the sequential results.

## Decision and next kernel gate

Decision: **reject for production; retain as a negative research result**.

## Packed-NF4 Triton follow-up

A second prototype removed the dense dequantized weight from the frozen-base
input-gradient calculation. It decodes bitsandbytes' high-nibble-first NF4
layout and resolved double-quantized block scales inside a tiled Triton
`dY @ W` kernel. Tests match the bitsandbytes dequantize-plus-matmul reference,
including dimensions that are not tile or quantization-block multiples.

The initial `16x32x32` tiling was approximately three times slower on the square
projection. Increasing the persistent M tile reduced repeated NF4 decoding. The
best valid `512x32x16`, four-warp geometry measured:

| Projection `(in -> out)` | Reference median | Packed candidate | Change |
|---|---:|---:|---:|
| `1536 -> 1536` | 0.8125 ms | 0.8631 ms | 6.2% slower |
| `1536 -> 8960` | 2.7889 ms | 3.9460 ms | 41.5% slower |
| `8960 -> 1536` | 2.9225 ms | 4.0620 ms | 39.0% slower |

The candidate reduced temporary allocation but failed every speed gate. A wider
reduction tile exceeded the GPU's shared-memory limit and produced no benchmark.
This prototype is also isolated from production. A competitive follow-up likely
requires native CUDA/CUTLASS-style producer-consumer pipelines or architecture-
specific MMA layouts that decode a weight tile once and reuse it across several
output tiles. That work is materially larger than a Triton scheduling patch and
must start with a component gate rather than an all-model claim.

A new packed-NF4 kernel must satisfy all of the following before integration:

1. Consume the existing bitsandbytes NF4 nibbles and nested scale state without
   changing quantization semantics.
2. Use tensor-core-friendly tiled accumulation for the frozen-base `dX` path.
3. Avoid a dense BF16 dequantized-weight allocation.
4. Match the reference forward and gradients on odd/tail shapes and adversarial
   scale values.
5. Beat bitsandbytes on square, up and down projections, not just one geometry.
6. Then pass all-layer time, energy, VRAM, held-out NLL and thermal gates.

## Primary references

- QLoRA paper: https://arxiv.org/abs/2305.14314
- bitsandbytes `matmul_4bit` implementation:
  https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py
- bitsandbytes CUDA build and 4-bit GEMM sources:
  https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/CMakeLists.txt
- Unsloth repository: https://github.com/unslothai/unsloth
