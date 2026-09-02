# One-billion-parameter / ten-minute feasibility analysis

## Claim under test

Train a 1B-parameter model from random initialization on the audited RTX 4060
Laptop GPU (8 GiB) and 16 GiB RAM in under 600 seconds.

The word “train” is split into non-interchangeable gates:

1. **Mechanism gate:** instantiate at least 1,000,000,000 unique trainable
   parameters, compute finite loss and gradients, apply at least one update, and
   stay within memory/time limits.
2. **Memorization gate:** substantially reduce loss on a fixed tiny corpus and
   retain a checkpoint.
3. **Useful pretraining gate:** improve preregistered held-out language metrics
   against smaller compute-matched models on nontrivial data.

Passing gate 1 is not evidence for gates 2 or 3.

## Optimistic compute bound

Dense Transformer training is commonly approximated as `6 × parameters ×
tokens` FLOPs, before attention and operational overhead. NVIDIA lists 233 “AI
TOPS” for the RTX 4060 Laptop GPU; that is not a sustained BF16 training rate,
but using it as if it were produces an intentionally generous upper bound:

```text
available compute = 233e12 × 600 = 1.398e17 operations
maximum tokens     = 1.398e17 / (6 × 1e9) = 23.3 million tokens
```

Compute-optimal scaling work trained models over billions of tokens and found
model size and training-token count should scale together. A conventional
20-tokens-per-parameter planning value would be 20 billion tokens for 1B
parameters. Reaching that in 600 seconds would require an impossible sustained
200 petaflop/s under the same approximation—about 858× the already generous
233-TOPS assumption, before overhead.

Primary sources:

- Hoffmann et al., *Training Compute-Optimal Large Language Models*:
  <https://arxiv.org/abs/2203.15556>
- DeepSeek LLM discussion of `6ND` and more exact attention-aware accounting:
  <https://arxiv.org/abs/2401.02954>
- NVIDIA RTX 40 Laptop specifications:
  <https://www.nvidia.com/en-us/geforce/laptops/compare/>

## Memory bound and selected mechanism baseline

`configs/one-billion-mechanism.json` defines a real dense model:

- 20 Transformer layers, width 2,048, 16 heads, MLP ratio 4;
- byte vocabulary and context length 8;
- exactly 1,007,710,208 unique parameters;
- BF16 parameter storage: 2,015,420,416 bytes;
- BF16 gradients: approximately the same again;
- activation checkpointing;
- SGD without momentum or master weights;
- one micro-batch, one optimizer step, 540-second internal deadline;
- no final checkpoint, because a ~2 GiB save would conflate mechanism compute
  with persistence and is tested separately.

This intentionally sacrifices optimizer quality and BF16 update resolution to
test the minimum-memory dense path. AdamW's two FP32 moments alone require about
8 GiB for 1B parameters, before weights, gradients, activations, and allocator
workspace, so it cannot be the honest 8 GiB baseline.

## Closest prior art

- ReLoRA reports training models up to 1.3B with repeated low-rank updates and
  memory/speed improvements: <https://arxiv.org/abs/2307.05695>.
- GaLore reports low-rank gradient projection and reduced optimizer memory:
  <https://arxiv.org/abs/2403.03507>.
- APOLLO reports SGD-like optimizer memory through low-rank random projections:
  <https://arxiv.org/abs/2412.05270>.
- Switch Transformers establish sparse conditional computation, not dense
  training equivalence: <https://arxiv.org/abs/2101.03961>.
- Cramming studies useful single-GPU training over a full day, not ten minutes:
  <https://arxiv.org/abs/2212.14034>.

These methods motivate comparisons; they do not validate the ten-minute claim.

## Experimental decision

Run the dense mechanism gate before inventing a sparse or low-rank substitute.
If it fits, record achieved tokens/s and calculate the measured—not advertised—
time to 1B and 20B tokens. If it OOMs, retain the failure and test one variable at
a time: shorter context cannot go below the current 8; next candidates are
selective gradient storage, low-rank updates, then conditional experts.

Any sparse model must report total parameters, active parameters per token,
trainable parameters per step, and FLOPs separately. It may not be compared to a
dense model using total parameter count alone.
