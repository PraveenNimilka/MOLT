# Prototype results — 2026-09-01

## Outcome

The `ai-local` prototype is operational on native Windows with Python 3.12.13,
PyTorch 2.8.0+cu128, and the RTX 4060 Laptop GPU. It implements strict configs,
deterministic byte preparation, CPU/CUDA training, AMP/BF16, activation
checkpointing, telemetry, atomic checkpoint metadata, previous-checkpoint
recovery, evaluation, comparison, reports, and optimistic feasibility bounds.

No generally useful 1B model was trained. EXP-1005 passed only a repeated-corpus
memorization gate.

## Runtime validation

| Check | Result |
|---|---|
| Automated tests | 14 passed |
| CPU smoke | 4 steps; validation 7.957→7.214 bits/byte |
| CUDA smoke | 10 steps; validation 8.003→4.175 bits/byte |
| BF16 GEMM ceiling | 24.31 TFLOP/s, 4096² matrices; not training throughput |
| CUDA smoke framework peak | 84.1 MB allocated |
| CUDA smoke GPU energy | 28.66 J sampled board energy |
| Data corruption | Rejected by SHA-256 validation |
| Checkpoint corruption | Latest invalid checkpoint rejected; valid previous generation selected |
| Completed-run resume | Rejected |

## One-billion-parameter results

All 1B runs used exactly 1,007,710,208 unique trainable parameters: 20 layers,
width 2,048, 16 heads, MLP ratio 4, BF16 weights, checkpointed activations,
context length 8, micro-batch 1, and a repeated synthetic byte corpus.

| Experiment | Steps / tokens | Final B/B | Time | Framework peak | Device VRAM peak | GPU energy | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| EXP-1001 direct SGD mechanism | 1 / 8 | 8.192 | 1.38 s | 4.05 GB | 5.42 GB | 23.4 J | Mechanism passed only |
| EXP-1002 SGD lr 1e-4 | 256 / 2,048 | 8.563 | 32.28 s | 4.05 GB | 5.42 GB | 2.07 kJ | Rejected: stagnation |
| EXP-1003 SGD lr 1e-2 | 128 / 1,024 | 5.934 | 16.92 s | 4.05 GB | 5.39 GB | 1.04 kJ | Retained control |
| EXP-1004 residual SGD | 128 / 1,024 | 4.490 | 31.92 s | 6.13 GB | 7.50 GB | 2.05 kJ | Not promoted |
| EXP-1005 persisted memorization | 2,048 / 16,384 | 2.922 | 256.73 s | 4.05 GB | 5.39 GB | 17.22 kJ | Narrow gate passed |

EXP-1005 time includes writing and hashing a 2,015,516,159-byte checkpoint. The
checkpoint SHA-256 is
`6057837dcb2245b555d12f0ed97228c9785c7727a0f88337b996e8a0183f6751`
and it independently reloaded to the same 2.9215925 validation B/B.

## Error-feedback comparison

ResidualSGD is a simple Kahan/error-feedback reproduction, not a new method.
Against the equal-learning-rate EXP-1003 control:

- final B/B improved 24.3%;
- interpolation estimates 372.7 vs 1,024 tokens to the control's final quality
  (2.75× sample efficiency);
- interpolated time-to-quality was 12.46 vs 16.92 seconds (1.36×);
- full-run throughput fell to 53.0% of control;
- framework peak memory grew by about 2.08 GB;
- full-run sampled energy nearly doubled.

It misses the pre-registered 2× end-to-end gate and is not promoted.

Closest prior work includes BF16 Kahan/stochastic rounding
<https://arxiv.org/abs/2010.06192> and ECO error compensation
<https://arxiv.org/abs/2601.22101>.

## The ten-minute claim

The measured steady training rate was approximately 64 tokens/s for the tiny
context workload. Ten minutes would process only about 38,000 tokens, versus a
planning target on the order of 20 billion for compute-optimal 1B pretraining.
The present evidence therefore supports:

- **yes:** instantiate, update, memorize a trivial repeated distribution, and
  persist a 1B checkpoint on 8 GiB in under ten minutes;
- **no:** train a generally useful 1B language model from scratch in ten minutes.

The second claim remains contradicted by both measured throughput and the
optimistic compute bound. Sparse total-parameter inflation, weight tying, or a
tiny memorization task must not be used to relabel it as achieved.

## Remaining uncertainty

Runs use one seed, background desktop GPU load, a tiny same-distribution
validation set, and only GPU-board energy. The pinned TinyStories baseline,
three-seed reproduction, deterministic CUDA resume tolerance, OOM injection,
and whole-system wall-energy measurement remain outstanding.
