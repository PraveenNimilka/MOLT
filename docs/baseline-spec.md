# Baseline specification — BL-0001

## Purpose and status

BL-0001 is the smallest workload intended to validate the experiment platform
and establish time/energy/memory-to-quality curves. It is **specified, not yet
run**. No baseline metric may be filled from expectation or another machine.

## Workload

| Field | Pre-registered value |
|---|---|
| Task | Autoregressive next-byte prediction from scratch |
| Dataset | `roneneldan/TinyStories`, train and validation, pinned Hub commit required |
| License | CDLA-Sharing-1.0 per dataset card; preserve attribution and terms |
| Subset | First 20,000,000 normalized UTF-8 training bytes in pinned source-row order; first 2,000,000 validation bytes in pinned source-row order |
| Normalization | UTF-8 encode; normalize CRLF to LF; concatenate records with one LF; no Unicode normalization |
| Integrity | Record source revision, raw/prepared SHA-256, row counts, byte counts, and overlap check |
| Vocabulary | Fixed 256 byte values; no trained tokenizer |
| Context | 256 bytes |
| Model | Decoder-only Transformer, pre-norm, learned position embeddings, GELU MLP, tied input/output embeddings |
| Shape | 8 layers, width 256, 8 heads, MLP width 1024 |
| Parameters | 6,449,664 expected; runner must compute and assert exact count |
| Attention | PyTorch `scaled_dot_product_attention`, causal; record selected backend |
| Dropout | 0.0 |

Preparation stops reading each split once the byte cap is reached, so the first
baseline does not require downloading or sorting the full 7+ GiB repository.
The resulting source-order bias is fixed across candidates and explicitly limits
generalization. The fixed subset is intentionally small and synthetic. It is suitable for
mechanism checks and is not representative evidence about broad language-model
quality.

## Optimization

| Field | Value |
|---|---|
| Optimizer | AdamW |
| Betas / epsilon | (0.9, 0.95) / 1e-8 |
| Weight decay | 0.1, excluding bias, embeddings, and normalization parameters |
| Peak learning rate | 6e-4 |
| Schedule | 100-step linear warm-up, cosine decay to 6e-5 |
| Token budget | 50,000,000 training bytes/tokens |
| Micro-batch | 16 sequences (4,096 tokens) |
| Accumulation | 4 micro-batches; 16,384 tokens per optimizer step |
| Precision | CUDA FP16 autocast with dynamic gradient scaling; CPU FP32 |
| Gradient clipping | Global norm 1.0, measured before clipping |
| Seeds | 1337, 2027, 3407 |
| Data order | Stateless permutation derived from run seed; stored sampler position |

The final step consumes the remaining budget rather than exceeding 50M tokens.
OOM is a recorded failure; the runner must not silently alter batch size,
context, precision, or accumulation.

## Evaluation and quality

- Evaluate before training, every 250 optimizer steps, and at the final step.
- Use the same fixed validation windows for every run, recorded by manifest hash.
- Primary metric: validation bits per byte (B/B), aggregated by total negative
  log-likelihood and token count—not a mean of batch means.
- Secondary metrics: validation NLL, train NLL, non-finite count, and optionally
  fixed-prompt samples clearly separated from quantitative evaluation.
- Do not use an LLM judge for the primary gate.

After all three baseline seeds complete, define the candidate target quality as
the worst (highest) final B/B among successful baseline seeds. Compute each
run's first time/energy crossing by linear interpolation between adjacent
evaluations. This target is frozen in a versioned comparison manifest before
running a candidate.

## Measurement protocol

Every run records immutable config, Git revision (or explicit `unversioned`),
dependency lock hash, dataset hashes, host fingerprint, seed, start/end times,
and raw phase samples.

Phases: startup, data-open, model-init, warm-up, training, evaluation,
checkpoint-save, checkpoint-load, and total process. Synchronize CUDA around
timed GPU phase boundaries. Warm-up uses an untimed compile/allocator exercise
without advancing model, optimizer, scaler, sampler, or RNG state.

Primary measurements:

- wall-clock and time to target quality;
- tokens/s excluding and including evaluation/checkpoint overhead;
- process working set and system available RAM samples;
- PyTorch peak allocated/reserved VRAM plus NVML used-VRAM samples;
- gross GPU board energy from integrated sampled power, sampling metadata, gaps,
  and telemetry overhead; optional idle-adjusted value is secondary;
- GPU/CPU/storage utilization samples where supported;
- checkpoint bytes and save/load duration;
- recovery equivalence and numerical failures.

The NVIDIA power reading is a board-level, approximately 1-second average with
documented accuracy limits. It is not whole-system energy. If sampling coverage
is inadequate, energy is `unavailable`, never guessed.

## Baseline acceptance gate

BL-0001 becomes trustworthy only if:

1. CPU smoke training passes and loss decreases on a tiny overfit fixture.
2. CUDA smoke training passes with no non-finite values.
3. uninterrupted vs checkpoint/resume runs match the pre-registered tolerance:
   exact sampler/step/token position and max parameter absolute difference
   ≤1e-6 on CPU FP32; CUDA tolerance is established by a separate deterministic
   reference test and never relaxed after observing candidate results.
4. all three seeds complete the exact token budget;
5. fixed-seed 200-step performance repetitions (five after one warm-up) have
   training-throughput coefficient of variation ≤5%, or the noise source is
   diagnosed and reported;
6. raw metrics validate against their schema and every phase duration reconciles
   with total time within telemetry overhead;
7. no training process exceeds 12 GiB RAM, 7.0 GiB steady VRAM, or the 20 GiB
   prepared-data/artifact budget.

Milestone 1 is not satisfied merely because one training command exits zero.

## Candidate success and regression gate

At equal or better target validation B/B, a candidate must show at least one:

- ≥2.0× lower median time-to-quality;
- ≥2.0× lower median measured GPU-energy-to-quality;
- ≥50% lower peak VRAM or RAM;
- ≥50% lower modeled local experiment cost using a published formula; or
- materially improved completion/recovery rate in a pre-registered fault test.

Unless the hypothesis says otherwise, reject promotion if median time or GPU
energy regresses >20%, final B/B worsens >1%, any seed fails to cross target,
non-finite updates occur, checkpoint bytes grow >25%, or storage traffic grows
>50%. Report bootstrap 95% confidence intervals and all individual runs; three
seeds are an initial gate, not universal proof.

## Environment prerequisite

Milestone 1 should create a repository-local `uv` environment using a supported
Python version and an official CUDA-enabled PyTorch wheel, then pin exact
versions in a lockfile. Installation and dataset download were deliberately not
performed in Milestone 0.
