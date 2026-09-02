# Milestone 3: isolated optimization research plan

## Decision

The first candidate is **shape-stable Windows Inductor fusion without CUDA
Graphs**. This is an established PyTorch mechanism, not a MOLT invention. It is
selected because the local kill test showed a large cached-step time signal and
a 14.37% incremental-memory reduction, while preserving the same mathematical
model and AdamW update. Its thermal, energy, cold-start, convergence and
repeatability behavior remain unproven.

MOLT will not combine layer streaming, GaLore, compiler fusion, power control or
new data sampling in this experiment. One independent variable changes:
`eager` versus `compile-max-autotune-no-cudagraphs`.

## Immutable workload

- Model: current 10,776,640-parameter RMSNorm/SwiGLU/SDPA decoder for the cheap
  kill stage; surviving code advances unchanged to the 49,883,520-parameter
  context-512 baseline.
- Data: identical prepared TinyStories BPE-8192 token files.
- Geometry: context 512, microbatch 24, accumulation 1, contiguous packing.
- Optimizer: fused AdamW with FP32 parameters and optimizer state.
- Seeds: 1337, 2027, 4099.
- Same initialization, token order, validation batches and token/update budget
  within each paired seed.
- Hardware: RTX 4060 Laptop GPU 8 GB on AC power. A pair starts only in the same
  preregistered temperature bin; execution order alternates AB/BA.

## Measurements

Two cost scopes are reported and never mixed:

1. Cold-run cost includes compilation, autotuning, warm-up, training,
   evaluation and checkpointing.
2. Warm-cache cost starts a fresh process with the existing compiler cache and
   still includes model initialization, warm-up, training, evaluation and
   checkpointing.

Primary outcome is seconds and NVML GPU-board joules to interpolated target
validation NLL. Secondary outcomes are peak allocated/reserved VRAM, process
RSS, samples/tokens per second, maximum temperature, throttle reasons,
checkpoint bytes and interruption recovery. CPU and wall-plug energy remain
unmeasured and are labeled exclusions.

## Preregistered gates

The candidate advances only when all conditions hold:

- final validation NLL is no more than 1% above the paired eager result;
- median time-to-target improves by at least 1%;
- median board-energy-to-target improves by at least 1%;
- peak allocated VRAM does not regress by more than 1%;
- peak temperature is at most 70 C with no thermal abort or throttle bit;
- every seed is finite and completes; checkpoint/resume remains equivalent;
- 95% paired bootstrap intervals for time and energy exclude zero regression.

Strong evidence is at least 25% joint time/energy improvement. The 50% target
remains a breakthrough gate, not an expectation.

## Cheap kill ladder

1. **Numerical correctness:** compare eager/compiled logits, loss and gradients
   on fixed mini-batches; run 20 identical updates and bound parameter drift.
2. **Graph/guard audit:** require one compiled graph, no silent eager fallback,
   no unexpected recompilation and finite BF16 SDPA behavior.
3. **Thermal block:** alternate five-update eager/compiled blocks from a common
   checkpoint; reject if compiled energy, temperature or loss progress regresses.
4. **Three-seed short convergence:** measure target NLL over a fixed small token
   budget with paired order.
5. **49.88M reproduction:** only a survivor moves to the immutable larger
   workload and full time/energy-to-quality analysis.

## Next candidates if rejected

Candidates are sequential, not combined:

1. Chunked linear-cross-entropy with exact loss/gradient equivalence. Liger is
   direct prior art, so this tests portability/value rather than novelty.
2. Static activation-lifetime-guided fusion partitioning that avoids the known
   Windows CUDA-Graph launcher failure. Novelty is unclaimed until a dedicated
   compiler/prior-art search is complete.
3. APOLLO versus AdamW as a memory-constrained optimizer experiment, with fresh
   hyperparameter fairness tests. It is established prior art and cannot be
   compared using AdamW settings without tuning both methods.

Layer bundling is excluded because its three-pair median gain was only 4.11%
and energy was mixed. FlashAttention-3 is excluded because its published
mechanisms target Hopper-specific TMA/warp-specialization rather than Ada.

## Reproduction entry points

```powershell
uv sync --extra windows-fusion
.\.venv\Scripts\molt.exe inspect
.\.venv\Scripts\molt.exe fusion-benchmark --config configs\molt-stream-production.json
.\.venv\Scripts\molt.exe benchmark --config configs\molt-stream-production.json --warmup-steps 10 --steps 30
```

The paired time-to-quality runner and numerical-drift gate are the next
implementation increment. No Milestone 3 promotion occurs until those artifacts
exist and pass the preregistered thresholds.
