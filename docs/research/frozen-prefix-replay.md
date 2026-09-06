# Frozen-prefix replay: exact-window caching screen

Status: experimental falsification study, not a release feature or novelty claim.
This study continues the 1.54B Qwen2.5 NF4 adapter experiments. Its hypothesis
was stated before the first trial: repeated token windows can reuse an unchanged
lower decoder representation while preserving upper-adapter updates. Cache
construction, host copies, misses, memory, validation and cooling must be counted.

## Prior art and scope

[AutoFreeze (2021)](https://arxiv.org/abs/2102.01386) combines layer freezing
and activation caching for faster single-GPU fine-tuning.
[PipeTransformer (2021)](https://arxiv.org/abs/2102.03161) includes automated
caching alongside elastic training; see the authors' account in the
[PyTorch article](https://pytorch.org/blog/pipetransformer-automated-elastic-pipelining/).
MOLT's implementation was written locally, but the underlying mechanism is
established prior art, not an original training algorithm.

This is **not** a shortcut for processing unseen data, full-parameter pretraining,
or changing lower-layer adapters. It applies only after a prefix is frozen and
identical input windows recur. Choosing that adaptation method can itself lose
quality relative to full-adapter tuning; caching equivalence does not resolve
that separate tradeoff.

## Mechanism and cost model

Write the decoder as `h = F_theta(x)`, `z = G_phi(h)`, where theta is frozen.
For deterministic F, storing h exactly preserves the loss and gradient with
respect to phi. No gradient through theta is required. MOLT caches the original
tensor dtype, not a quantized approximation, and retains the usual upper-layer
backward pass and FP32 AdamW.

For N distinct windows repeated E times, let F be frozen-prefix forward time,
U the remaining forward/backward/update cost per window, W the cache write cost,
R the read cost, and H total lookup/validation overhead. Ignoring thermal
feedback only for this analytical estimate:

```
T_uncached = E N (F + U)
T_cached   = N (F + W) + E N U + (E - 1) N R + H
gain requires (E - 1) N (F - R) > N W + H
```

Actual acceptance uses measured session time and board joules, not this estimate.
Tensor storage is `N * batch * context * hidden_width * element_bytes`.
Sixteen B1/L512/width1536 FP32 windows require 48 MiB. A full 4.1M-token
dataset would require roughly 25 GB in FP32, so indiscriminate caching would
violate the laptop memory budget. Bounded LRU caching can instead have zero
reuse across long sequential epochs: that outcome must count as failure.

## Implementation boundaries

`training/frozen_prefix.py` is an experimental context manager, not wired into
the public CLI. It temporarily bypasses frozen decoder blocks on exact-window
hits and restores all original forwards on exit, including exceptions.

- Only Qwen2 tensor-returning decoder blocks, fixed default RoPE, explicit
  unmasked token windows, and zero prefix dropout are accepted.
- The embedding and prefix must have no trainable parameters.
- Parameter/buffer versions, tensor identities and model configuration are
  checked before reuse. Ordinary weight updates invalidate the experiment with
  an explicit error. Direct `.data` mutation and concurrent forwards are unsupported.
- Cache keys include complete token bytes, shape, device, training mode, AMP
  mode/dtype and the selected matmul/SDPA precision backends. Prefix outputs
  retain their exact dtype on CPU.
- Tensor bytes are bounded; Python keys and bookkeeping add overhead. Process
  RSS is measured separately. This is not a zero-RAM claim.
- The cache is ephemeral. A fresh/resumed process must pay cold-fill costs again.
  This prototype does not claim checkpoint/resume integration.

CPU correctness checks cover exact outputs, trainable gradients and RNG with
checkpointing both enabled and disabled, bounded eviction, stale-prefix rejection,
dropout/trainability rejection, and restoration after an unsupported-mask error.

## Protocol and acceptance

Use the official Qwen2.5-1.5B checkpoint (1,543,714,304 base parameters), rank-8
adapters in the last three decoder blocks, context 512, B1/G4, LR 2e-4, and the
same frozen-head loss and fused FP32 AdamW in both arms. Keep the exact same
window sequence and epoch count. The first screen uses a repeated small subset,
not the complete dataset; that limitation is intentional and explicit.

Cheap gates: both arms complete; matching loss trajectories and final held-out
NLL within 1%; lower measured session time and board joules; no worse peak VRAM
or thermal behavior. A pass only permits larger multi-seed/dataset studies.
It cannot promote a method or establish a competitive advantage by itself.

The disposable benchmark includes start cooldown, model initialization, initial
and final validation, cold cache fill, transfers, all attempted updates and pacing.
Imports, process teardown and JSON serialization are excluded and labeled.
No CPU-temperature sensor is available. GPU software checks cannot preempt an
already running kernel. The optional forward guard checks between decoder blocks.

## Initial results (preserved, not selected away)

Artifacts live under `artifacts/mars-1p5b-20260904/`.

- `replay-cache-16w4e-seed1337.json`: completed 16 updates / 32,768 tokens;
  3,318.83 retry-inclusive compute tok/s, **591.17 session tok/s**, 55.4292 s,
  approximately 1,623.79 board J, 2.58 GiB allocated VRAM, 48 MiB cached tensors,
  48 hits / 16 misses, 38.8779 s pacing, and a 71 C gate peak. One-window NLL
  improved from 1.743079 to 1.651907. CPU unit tests overlapped part of this run,
  creating a background-load confound; do not treat it as a controlled speed result.
- `replay-baseline-16w4e-seed1337.json`: stopped at 72 C during its first update,
  with zero committed tokens and 1,024 discarded tokens. No final quality result.
  This pair is **inconclusive**, not a caching speed or quality victory.

The next paired protocol adds the same between-layer forward thermal guard to
both arms and requires a starting GPU sample at or below 50 C. It preserves the
60 C pacing and 72 C stop boundaries. No temperature threshold is raised.

### Guarded three-seed screen

All six fresh-process arms completed eight updates / 16,384 predicted tokens.
Each used eight unique 512-token windows repeated four times. For each seed,
every recorded training loss and the final one-window NLL matched exactly
between baseline and cache. CPU tests did not overlap these six trials.

| Seed | Baseline seconds | Cache seconds | Baseline board J | Cache board J | Cache compute tok/s | Cache session tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1337 | 20.3578 | 15.5994 | 813.99 | 556.70 | 3583.76 | 1050.29 |
| 1338 | 24.1290 | 15.0100 | 874.64 | 531.07 | 3682.15 | 1091.54 |
| 1339 | 21.7892 | 31.9006 | 866.61 | 757.95 | 3590.34 | 513.60 |

All arms peaked at 2,769,402,880 allocated GPU bytes (2.58 GiB). Cached arms
held 24 MiB of CPU tensors, had 24 hits / 8 misses, and peaked at 68 C according
to the synchronous guard. Baseline guard peaks were 69, 70 and 70 C. Process
RSS is **not** the cache size: for seed 1337 it was about 3.54 GiB baseline
versus 3.57 GiB cached. Other seeds' RSS deltas varied; see raw telemetry.
CPU temperature is still unavailable.

The first two pairs reduced total time by 23.37% and 37.79% and board joules
by 31.61% and 39.28%. The third reduced board joules by 12.54% but increased
total time by 46.41%. Its setup, including initial cooldown, took 22.21 seconds
versus 5.75 seconds for the baseline. That cost remains included; removing it
after seeing the result would change the registered session metric.

`replay-screen-three-seeds.json` automatically returns **reject-or-inconclusive**
and `milestone_passed: false`. The representation-reuse correctness hypothesis
survived; a repeatable joint session-time/energy improvement did not. No confidence
interval or independent-reproduction claim is justified by three short pairs.
The one-window held-out NLLs were respectively 1.705593, 1.703420 and 1.700742;
these do not establish parity with the stronger full-adapter target near 1.575.

Next experiment: enlarge the working set beyond the cache budget and preserve
complete-dataset order. Measure hit-rate collapse and total cost before adding
this to the public runner. Separately repeat paired thermal trials with a
predefined idle-stability protocol and report both preconditioning cost and
training-session cost; never silently remove the former. Public integration
also requires resume equivalence and a larger held-out evaluation.

## Reproduction

From the repository, using its environment rather than a globally installed MOLT:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_molt_stream_frozen_prefix.py tests/test_molt_stream_update_accounting.py
.\.venv\Scripts\python.exe benchmarks/frozen_prefix_replay.py --config artifacts/mars-1p5b-20260904/guard-upper3-fast-v1.json --forward-guard --windows 8 --epochs 4 --output artifacts/mars-1p5b-20260904/replay-guarded-baseline-8w4e-seed1337.json
.\.venv\Scripts\python.exe benchmarks/frozen_prefix_replay.py --config artifacts/mars-1p5b-20260904/guard-upper3-fast-v1.json --forward-guard --windows 8 --epochs 4 --cache --output artifacts/mars-1p5b-20260904/replay-guarded-cache-8w4e-seed1337.json
.\.venv\Scripts\python.exe -m molt_stream.experiments.replay_screen --root artifacts/mars-1p5b-20260904 --output artifacts/mars-1p5b-20260904/replay-screen-three-seeds.json
```

Choose new output filenames: existing artifacts are never overwritten. The
result embeds the resolved training spec, probe arguments, package versions,
benchmark source hash, per-update timings/losses, NLL and raw NVML samples.
The local artifact config is not a packaged production preset. On another
machine, reconstruct it from the embedded `spec` and change only model/data
paths to the same verified assets; record those differences. Dataset preparation
and model identity are documented in `soup-unsloth-thermal-study.md`.

For seeds 1338 and 1339, append `--seed 1338` or `--seed 1339` and use matching
output filenames before running the three-seed summary. Seed 1338 ran cached
first; seeds 1337 and 1339 ran baseline first. Later probe runs also save a
`.source.py` snapshot and exception traceback. All failures remain artifacts.

Verification at the end of this cycle: full `pytest -q` passed 223 tests.
After the final precision-key/provenance checks, all 18 focused replay,
accounting and screening tests passed. `git diff --check` passed. No release,
public CLI integration, Git commit or push was performed in this cycle.
