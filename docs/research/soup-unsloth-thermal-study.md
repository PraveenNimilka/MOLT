# MOLT: 1.5B thermal safety and speed study

**Status:** research result; not a Milestone 3 promotion  
**Date:** 2026-09-04

## Outcome

MOLT now protects gradient-accumulated QLoRA updates transactionally. It checks a fresh GPU-temperature sample around each forward and backward operation and during validation. If temperature or telemetry becomes unsafe before `optimizer.step()`, MOLT drops the partial gradients, restores the memory-mapped data cursor and random-number states, then writes an atomic checkpoint containing only committed work. Tests cover CPU rollback and interruption/resume equivalence on a real, tiny NF4 LoRA model.

This fixes correctness and recovery; it does **not** create the missing speed. The registered Qwen2.5-1.5B experiment has not reached 1,500, 1,800, or 2,500 sustained end-to-end tokens/s under the thermal gate.

## What the architecture research says

Soup's pinned runtime uses layer prefetch and non-reentrant recomputation to exchange resident VRAM for repeated PCIe reads and compute. Frozen weights are still required to calculate activation gradients. Therefore, streaming is useful when capacity is the blocker, but it is unlikely to speed up this 1.5B workload, which already fits in 4-bit form. [Soup source at the reviewed revision](https://github.com/MakazhanAlpamys/Soup/blob/766f348bfb727bc22aae0f849ef4180a851b987f/src/soup_cli/utils/layer_stream_runtime.py).

Unsloth's pinned fast-LoRA path uses model-specific manual autograd, fused elementwise MLP derivatives, and direct LoRA/base matrix operations under restrictions such as compatible bias, dropout, and adapter modes. Its vocabulary-loss code already uses adaptive exact chunking. These are strong engineering references, but exact loss partitioning is established work—not a MOLT invention. [Unsloth fast LoRA](https://github.com/unslothai/unsloth/blob/a2a4143ce69f6f7ddb13ed716f6b39756c398705/unsloth/kernels/fast_lora.py), [Unsloth Zoo loss](https://github.com/unslothai/unsloth-zoo/blob/4a1f75db0c3f0593edf16c360308380be253415e/unsloth_zoo/fused_losses/cross_entropy_loss.py), and [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009).

Predictive coding is a legitimate longer-term local-learning direction, but current evidence does not show that it preserves the registered AdamW/QLoRA objective or accelerates Qwen on an RTX 4060. It belongs in a separate algorithm comparison, not inside this baseline. [Millidge, Tschantz, and Buckley (2020)](https://arxiv.org/abs/2006.04182).

## Fixed experimental contract

The workload uses the official Qwen2.5-1.5B checkpoint at revision `8faed761d45a263340a0528343f099c05c9a4323`: 1,543,714,304 logical parameters, NF4 QLoRA rank 8 over all linear layers, AdamW, context 512, effective token batch 2,048, seed 1337, and a fixed tokenizer-prepared TinyStories split. The initial acceptance gate is at most 4 GiB allocated CUDA memory, a 60 C pacing target, a 72 C stop decision, and no more than 1% validation-NLL regression.

The data split is a contiguous tail of one source file. It is suitable for controlled local A/B tests but is not a contamination-free public quality benchmark. CPU package temperature is not exposed by the available telemetry, so this study makes no CPU-temperature claim.

## Results

| Configuration | Committed updates | End-to-end tok/s | Compute-only tok/s | Allocated peak | GPU peak | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full loss, activation recomputation | 4 | 402.26 | 1,037.17 | 3.05 GiB | 77 C | stopped safely |
| Exact partitioned loss, recomputation | 4 | 393.55 | 933.66 | 2.44 GiB | 73 C | stopped safely |
| Partitioned loss, no recomputation | 0 | — | — | 1.94 GiB | 75 C | hot-start refusal; invalid speed trial |

Run artifacts are under `artifacts/mars-1p5b-20260904/microguard-runs/`. The partitioned path lowered allocated peak by about 20% in the two comparable short runs. It did not improve speed. Both arms stopped before equal-token completion, so they cannot establish quality or energy superiority. Temperature can rise past the boundary during a single non-preemptible CUDA operation; “72 C” is consequently a stop-decision boundary, not a guaranteed physical maximum.

The instrumented trace identified matrix multiplication, tensor copies, NF4 dequantization, and base/LoRA GEMMs as major work. Profiler CPU attribution and native CUDA kernel rows overlap, and profiling itself substantially slows the run, so those observations guide optimization but are not benchmark percentages.

## Reproduce

From `${MOLT_REPO}` in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m molt_stream.training.profile_qlora `
  --config artifacts\mars-1p5b-20260904\guard-reference.json `
  --output artifacts\mars-1p5b-20260904\profile-new
.\.venv\Scripts\python.exe -m molt_stream.cli --json train `
  --config artifacts\mars-1p5b-20260904\guard-reference.json
.\.venv\Scripts\python.exe -m molt_stream.cli --json train `
  --config artifacts\mars-1p5b-20260904\guard-chunk.json
```

Begin each hardware arm only after the same idle-temperature condition. Keep all aborted runs. Do not compare compute-only speed with end-to-end speed, or runs that committed different token counts.

## Decision and next gate

The safety hypothesis survived. The current speed hypothesis did not, so Milestone 3 remains open. The next cheap, falsifiable test is a cold-start, equal-token comparison that uses the partitioned head's memory saving to disable selected activation recomputation. Only if operator-shape profiling shows enough removable work should MOLT add a narrowly scoped, model-specific LoRA backward kernel. Reaching 2,500 tok/s from the observed 1,314 tok/s compute-only path requires removing about 47% of total update time; small pacing or CLI changes cannot achieve that.

No reviewed mechanism is presented as novel, and no incomplete run is presented as a breakthrough.

## Follow-up: selective recomputation, 2026-09-05

Added experimental `qlora_checkpoint_stride` (default 1). A value of 2 enables
recomputation on alternating Qwen2 decoder layers, keeping the other activations.
The model, data order, batch, LoRA targets, and optimizer remain identical.
Unsupported or disabled layer checkpoint controls fail explicitly. A small Qwen2
test verifies loss and every parameter gradient against full checkpointing.

Run `20260905-100624-qlora-cec17236`, using `guard-selective.json`, committed
4 updates before stopping during validation. It measured 396.86 end-to-end tok/s,
1,067.20 compute-only tok/s, 3.76 GiB allocated peak, 74 C GPU peak, and
980.58 board joules. It passes the allocated-memory threshold but fails thermal
and throughput acceptance. No final quality comparison is available. These are
single-run diagnostics, not statistically demonstrated gains.

The earlier cold-start no-recomputation retry `20260904-234214-qlora-6e85331e`
committed 4 updates at 531.67 end-to-end tok/s and 1,247.01 compute-only tok/s,
with 5.06 GiB allocated peak and 77 C GPU peak. It fails the memory and thermal
thresholds. The driver also rejected `nvidia-smi -pl 65` as unsupported; no power
setting changed. Further speed claims require successful completed trials and
operator-level optimization, especially the validation vocabulary projection.

## Frozen vocabulary gradient experiment, 2026-09-05

Opt-in `qlora_precompute_head_gradient: true` requires partitioned QLoRA loss
and a frozen classifier. For each token chunk, the implementation computes loss
and its hidden gradient together, then retains only that gradient for backward.
This removes the old path's repeated vocabulary projection. For token count T,
hidden width H, vocabulary V and chunk C, saved state is O(TH), with temporary
logits O(CV). First-order gradients are supported; trainable heads are rejected.
This is an application of known gradient precomputation, not a novel algorithm.

CPU float64 tests cover uneven chunks and non-unit upstream loss scaling.
A BF16 CUDA diagnostic at T=512, H=1536, V=151936, C=128 produced identical
loss and maximum gradient difference 0. After the first iteration, old timings
were 52.85, 52.63, 52.99 ms; candidate timings were 36.37, 36.17, 33.49 ms.
This short sequential microbenchmark does not establish sustained gains.

The integrated run `20260905-103455-qlora-c605d884` committed 4 updates:
466.93 end-to-end tok/s, 1,164.00 compute-only tok/s, 3.77 GiB allocated peak,
73 C GPU peak, 886.46 board joules. It stopped during validation. Its resolved
configuration and measurements are preserved with the checkpoint. Different
initial conditions and incomplete validation prevent a quality or energy-win
claim against previous runs. The 2,500 tok/s acceptance gate remains open.

Reproduce the integrated arm with the saved `spec.resolved.json` in that run
directory. New runs must use a new output directory. Before promotion, perform
interleaved repeated microbenchmarks, CUDA AMP gradient tests across seeds,
complete equal-token model trials, and matched Soup/Unsloth baselines.

## Validation boundaries and completed recovery trial

Added opt-in `qlora_validation_layer_guard` for Qwen2. Temporary decoder hooks
check fresh temperature between validation layers and are removed on success or
exception. Tensor outputs match the unguarded path in a CPU test. This adds
synchronization overhead and cannot preempt running kernels.

Run `20260905-103956-qlora-882d231a` reached step 10 with NLL 1.661570 at
step 0, 1.557108 at step 4, and 1.536023 at step 8. Its first session consumed
45.418858 seconds and 2325.080696 board joules for 20,480 committed tokens;
1,536 uncommitted tokens were discarded on thermal stop at 72 C. Compute-only
rate was 1,157.97 tok/s, allocated peak 3.77 GiB. These first-session values
were captured from tool output before discovering that resume overwrote reports.

The second session committed the remaining 4,096 tokens in 11.831932 seconds
and 541.891907 board joules, but stopped at 73 C before final validation.
A recovery fix now permits pending validation at `step == max_steps` and
archives existing session reports before replacing them. The third session
completed validation (NLL 1.574904) and marked the run completed without further
optimizer updates. This is a recovered run, not uninterrupted thermal stability
or a 2,500 tok/s result. Earlier unarchived raw telemetry was overwritten by the
second session; only captured summaries survive, limiting reproducibility.

The separate synthetic precision probe (`head-precision-probe.json`) measured
8.06 ms versus 5.43 ms median warm projection/loss time for FP32 `highest`
versus `high`; NLL changed from 12.0755405 to 12.0755444. No production precision
setting was changed. This synthetic result requires real-model validation.

## MOLT truncated-adapter backpropagation experiment

The experimental `qlora_train_last_layers` option places adapters only in the
upper Qwen2 decoder layers and detaches the frozen lower representation at that
boundary. The lower decoder still runs forward; its activation graph and
backward work are removed. `qlora_full_warmup_steps` optionally begins with
adapters in every layer, then freezes lower adapters and activates the boundary.
The optimizer remains AdamW and only parameters with gradients update. This is
related to established layer-selective fine-tuning and is not currently claimed
as a novel algorithm.

Three fresh-process upper-three-layer trials measured compute-only
rates of 2678.35, 2594.95, and 2647.89 tok/s (mean 2640.40 tok/s). They used the
same official 1.54B checkpoint, context 512, 2048 tokens per optimizer update,
NF4 base, rank-8 adapters, fused FP32 AdamW, exact chunked loss, and seeds 1337,
1338, and 1339. They used unequal session lengths (12, 4, and 4 updates); the
12-update session thermally stopped before its registered endpoint. These are
observations, not a statistically established repeatability gate. Peak allocated memory
was 2.58 GiB in clean runs. GPU peaks were 70--72 C for the two short repeat
runs and 72 C for the longer seed-1337 session. The corresponding end-to-end
rates were 1203, 809, and 956 tok/s; these include setup, evaluation, and thermal
pacing and do not meet a 2500 tok/s end-to-end gate.

The one-layer ablation completed at 2257.26 compute tok/s, 2.57 GiB allocated,
59 C peak, and no thermal pause. Smaller backward work did not improve speed,
possibly consistent with loss of efficient GPU work granularity, but that cause
has not been established by profiling. Three layers are a promising tested
configuration, not a proven optimum.

The 32-update upper-three trial eventually reached four-window NLL 1.506833,
better than the earlier full-adapter step-12 four-window NLL near 1.575, but it
used more updates and total energy. It is not a time-to-quality win. A higher-LR
12-update arm completed in 27.14 seconds, used 1265.43 board joules, and measured
2617.86 compute tok/s. Its separately paced four-window NLL was 1.583876, within
about 0.57% of 1.574904 but not better.

A stronger full-adapter baseline using the same fused optimizer, head path, and
thermal recovery reached one-window NLL 1.565246 at step 4. It measured only
987.09 compute tok/s but reached target quality earlier. It exhausted eight
thermal recoveries at step 9 after 92.59 seconds and 3809.08 joules. Because the
baseline and candidate validation windows differ in some historical artifacts,
these runs do not establish a formal time-to-quality or energy-to-quality win.

The four-global-update then upper-three schedule reproduced the baseline's
one-window NLL 1.565246 at step 4 and improved it to 1.552606 at step 8. It
completed without thermal abort, with a 71 C peak. Post-switch updates were
faster, while whole-run compute averaged 1409.06 tok/s because it includes the
four global updates. This preserves quality relative to the switch point, not
relative to continued full-adapter training: full-adapter step-8 NLL was 1.511346,
so 1.552606 is about 2.73% worse and outside the 1% equal-update tolerance.
It does not demonstrate 2500 end-to-end tok/s.

Automatic `thermal_recovery_mode: cool-and-continue` now checkpoints committed
state, performs no CUDA work until a fresh sample reaches `thermal_recovery_c`,
and retries the rolled-back update without advancing data or RNG state. In the
32-step run it recovered eight times and reached step 24 in one process. Its
compute rate remained 2585.59 tok/s, while 38.37 seconds of passive cooling
reduced end-to-end throughput to 757.97 tok/s. A resumed session completed step
32. The measured peak reached 74 C because software cannot preempt an executing
CUDA kernel and NVML temperature updates are delayed.

### Current decision

The 2500 tok/s **successful-update compute path** has been observed on a 1.54B
model with three seeds. Unequal run lengths, thermal interruption, and omitted
discarded-update compute prevent treating that as the repeatability gate. The
combined 2500 tok/s end-to-end, low-temperature gate is not passed. Initial
Unsloth diagnostics do not yet establish a fair competitive advantage.
MOLT exceeds Soup's published 1.5B number only
across different hardware/software conditions, which is not a valid competitive
benchmark. The method remains experimental pending matched baselines, longer
runs, multiple datasets, and statistical time-to-quality comparisons.

## All-layer activation-retention screen, 2026-09-05

After rejecting upper-layer-only speed as a general all-layer comparison, MOLT
screened ordinary selective activation checkpoint placement. Every arm retained
all 9,232,384 LoRA parameters, the same Qwen2.5 1.54B base, seed 1337, token order,
2,048-token optimizer update, FP32 fused AdamW semantics, exact partitioned loss,
and four held-out windows. Only decoder checkpoint stride changed. This is an
established save-versus-recompute tradeoff, not a new algorithm.

| Checkpoint stride | Update tok/s | End-to-end tok/s | Allocated GiB | J/token | Peak C | Final NLL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1,190.24 | 381.38 | 4.02 | 0.1232 | 72 | 1.557046 |
| 4 | 1,298.43 | 506.07 | 4.67 | 0.0998 | 68 | 1.557046 |
| 8 | 1,378.59 | 479.38 | 4.95 | 0.1003 | 70 | 1.557046 |
| 16 | **1,432.60** | 487.70 | 5.14 | **0.0994** | 69 | 1.557046 |
| 32 | 1,412.35 | 445.81 | 5.23 | 0.1069 | 73 | 1.557046 |

Stride 16 is the provisional compute/memory knee and stride 32 is rejected. The
screen passes a 1,200 update-compute diagnostic, but **does not pass 1,200
end-to-end tok/s**. These four-update runs are dominated by setup, two four-window
evaluations, and thermal pauses. They are too short for late-run stability and
use one seed. Promotion requires interleaved stride-1/stride-16 runs across three
seeds and both AB/BA orders, starting from a registered temperature band, with
time and board energy measured to the same interpolated held-out NLL. A longer
run is not allowed to amortize overhead in one arm but not the other.

The official Unsloth implementation exposes compiled patches, specialized
gradient checkpointing, and cut-cross-entropy paths. Soup documents that its
layer streaming is a memory-fit mechanism and explicitly says layer streaming
itself is not new. MOLT therefore cannot claim novelty for checkpoint placement,
partitioned loss, or layer streaming. A defensible result must beat the strongest
compatible combination under the paired protocol above.

### Preregistered three-seed result

Artifact: `artifacts/mars-1p5b-20260904/stride-ab-20260905/results.json`.
Target NLL was fixed at 1.58 before execution. Runs used all 9,232,384 adapter
parameters, four held-out windows, context 512, B1/G4, and alternating A/B order.

| Seed | Order | S1 time/J to target | S16 time/J to target | S16 compute tok/s | Outcome |
| ---: | --- | ---: | ---: | ---: | --- |
| 1337 | S1 then S16 | 26.74 s / 1,017.80 J | 18.96 s / 765.46 J | 1,397.97 | Pair passed |
| 2027 | S16 then S1 | 17.13 s / 820.19 J | 13.28 s / 634.84 J | 1,400.78 | S1 thermal abort |
| 4099 | S1 then S16 | 17.94 s / 850.20 J | 17.16 s / 802.73 J | 1,397.63 | S1 abort; S16 peaked 75 C |

Final NLL changes were zero for seed 1337 and below 0.15% in the incomplete pairs,
but two pairs failed completion/thermal gates. Stride-16 is repeatable near 1,399
update-compute tok/s, yet **promotion is rejected**. The short-run end-to-end rates
were 378--512 tok/s for Stride-16, far below the requested 1,500 tok/s. Mean total
CPU utilization across all six processes was 9.3--11.4%, already below the 15%
screen; CPU package temperature was not observable. No global PyTorch thread cap
was added without evidence that it would improve time, energy, or thermals.

The next kernel experiment is preregistered in
[fused exact vocabulary loss](fused-exact-vocabulary-loss.md).

## Continuation audit, 2026-09-05

Compute accounting version 2 includes discarded partial-update work in
`update_compute_seconds` and its useful-token rate. Separate
`committed_update_compute_*` fields retain the successful-update diagnostic.
Older artifacts remain unchanged and use the old definition. End-to-end rates
already included retries and cooling.

The isolated competitor diagnostics used Unsloth 2026.9.2, Zoo 2026.9.1,
Transformers 5.5.0 and Torch 2.8.0+cu128. MOLT uses Transformers 5.16.1.
No competitor packages were added to MOLT's lock file. All runs used the official
1.54B Qwen2.5 checkpoint, seed 1337, context 512 and 2048 predicted tokens/update.

| Diagnostic (4 updates) | Compute tok/s | Allocated bytes | Board J | Peak GPU C |
| --- | ---: | ---: | ---: | ---: |
| Unsloth, full logits, all-layer adapters | 696.23 | 3923962368 | 396.11 | 64 |
| Unsloth, CCE with filtering disabled, all-layer adapters | 944.94 | 2854052352 | 375.96 | 65 |
| Unsloth, CCE, requested last 3 layers | 987.04 | 2829890048 | 304.65 | 58 |

Artifacts: `artifacts/mars-1p5b-20260904/unsloth-seed1337.json`,
`unsloth-cce-full-seed1337.json`, `unsloth-cce-upper3-seed1337.json`.
These are **not accepted matched benchmarks**: no common held-out evaluation,
different thermal checking granularity, no warm steady-state interval, different
framework versions, and no recorded trainable-tensor inventory. Unsloth's
`finetune_last_n_layers` also establishes that upper-layer adaptation is already
an available competitor feature. Requesting it alone does not verify identical
adapter placement or initialization. XFormers/FlashAttention2 were unavailable
in the successful diagnostic environment. A backend/installation limitation
is not a competitor algorithm failure.

A geometry screen used B2/G2, keeping token order and the 2048-token update.
Run `20260905-151818-qlora-8c36ae67` completed 8 updates in 14.1132 s:
1945.54 compute tok/s, 1160.90 end-to-end tok/s, 491.25 board J and 69 C.
One-window NLL was 1.743079 initially and 1.702806 finally. The B1/G4 control
`20260905-152014-qlora-086597a9` stopped after 4 updates at 74 C: 2661.28
successful-update compute tok/s but only 699.14 end-to-end tok/s. This does not
establish an equal-endpoint quality or energy comparison; doubling microbatch
did not improve the observed compute rate. Both resolved specs and failures
remain in the artifact directory.

The next isolated hypothesis and its negative results are in
[Frozen-prefix replay](frozen-prefix-replay.md). This uses established frozen
activation caching, not a new training algorithm or a single-pass data speedup.
