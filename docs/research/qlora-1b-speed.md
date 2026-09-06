# 1B+ QLoRA speed investigation

Date: 2026-09-04. Target: sustained >=2,500 training tokens/sec on an RTX4060
laptop, not 0.5B performance or inference speed. This is adapter fine-tuning,
not full-model pretraining. Initial screen: allocatedVRAM<=4GiB, GPUpeak<=72C,
NLL within1% of matched baseline. CPU temperature unavailable. No thermal or
performance guarantee is implied. User approved a separate pretrained download.

## Correct model first

Use official [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), revision
`8faed761d45a263340a0528343f099c05c9a4323`, Apache2.0 model card, with its
own tokenizer. The Test110 custom checkpoint has1,107,776,000 stored parameters,
40Q/K normalization tensors incompatible with its selectedQwen2 architecture,
and GPT2 special-token IDs inconsistent with its config. Its reexport script
initializes random weights rather than loading a trained checkpoint. Actual
checkpoint provenance remains unverified. No original model files were edited.

## Implemented isolated mechanisms

1. `qlora_autocast: true`: use BF16 autocast for forward GEMMs, with explicit
   FP32 cross-entropy reduction. NF4 compute_dtype alone does not make every
   adapter/output-head GEMM mixed precision. Training and generation use the
   resolved option; evaluation always disables autocast for a common comparison
   precision (NF4 base computation remains BF16). BF16 changes rounding; equivalence is empirical,
   not bit-exact or automatic. Defaultfalse preserves existing behavior.
2. `qlora_fused_optimizer: true`: request PyTorch fusedAdamW explicitly. Refuse
   non-CUDA or non-FP32 trainable parameters; no hidden cast/fallback. AdamW
   moments remainFP32. Defaultfalse. Applicable toQLoRA, includingLoRA+ kwargs.

These are established implementation optimizations, not new algorithms.
Tests verify configuration, BF16 GEMM/FP32 reduction and gradients, and fused
AdamW updates/moments versus the reference over10updates, and an actual tiny
NF4+LoRA model with autocast and non-reentrant recomputation. The current full
suite collects and passes **252 tests**; `git diff --check` also passes.

## Measured outcome: target not achieved

The downloaded model contains1,543,714,304 logical parameters. Verified SHA256:
`a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`.
The accelerated and Python HTTP downloads stalled; a resumed direct download
of the pinned revision completed and was verified before loading. No existing
model files were replaced. No Unsloth comparison was attempted.

| Arm | Completed updates before abort | Run tok/s | Loop tok/s | Peak allocated GiB | Peak GPU C |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2 | 555 | 991 | 2.62 | 73 |
| BF16 autocast only | 5 | 762 | 1,013 | 3.05 | 74 |
| Autocast, microbatch4 | 11 | 859 | 983 | 5.88 | 76 |
| Autocast, microbatch1, no checkpointing | 9 | 1,057 | 1,314 | 5.67 | 79 |

All four runs thermally aborted before their12-update budget. These are partial
run measurements, **not sustained throughput or equal-quality comparisons**.
Baseline has only its initial validation evaluation; other runs stopped at
different token counts. Therefore neither quality noninferiority nor joint
time/energy improvement can be inferred. The larger-batch and no-checkpointing
arms also exceed the initial4GiB allocated-memory screen. Fusedbatch4 was not
run after that geometry failed the memory/thermal screen.

The user authorized a temporary65W power experiment. NVML reports an enforced
140W limit and a5–140W range, but its configurable-power-limit getter returned
`Not Supported`. The experiment refused to change hardware without a reliable
original setting to restore. **No power limit was changed**; see `power65.json`.
Reported supported range is not proof that setting/verification works.

MOLT now checks fresh telemetry before and after each accumulated microbatch.
If a boundary is reached before an optimizer commit, the data cursor, RNG and
optimizer state roll back transactionally. Automatic checkpoint, passive cool
and retry preserves committed work. A delayed NVML peak is reconciled before a
run can be declared complete. These mechanisms passed focused recovery tests,
but a72C decision threshold is still not a physical maximum: an executing CUDA
kernel cannot be preempted and the sensor reports with delay. Raw sensor samples
and complete specs/checkpoints/logs remain under
`artifacts/mars-1p5b-20260904/`; no failed case was discarded.

Profiling isolated repeated FP32-to-BF16 conversion of Qwen's frozen vocabulary
head. A non-persistent BF16 compute cache removes that repeated467MB conversion
while the FP32 master remains available for validation. With all28layers active,
resident activations, exact analytical loss, batch1/accumulation4 and context512,
an eight-update run measured1,532 update-compute tok/s,5.32GiB peak PyTorch
allocation and68C, with NLL improving1.66157 to1.53397. It completed, but thermal
waiting and validation reduced end-to-end throughput to621tok/s.

A16-update trial with a64C target measured1,556 update-compute tok/s and759
end-to-end tok/s, but touched the72C boundary and was correctly marked as a
thermal abort. The custom Triton chunk-loss operator passed its component test
but was slower than analytical loss in matched full training; details are in
`docs/research/fused-exact-vocabulary-loss.md`. Therefore the1,500 compute rung
is achieved once with the production analytical backend, while the sustained
1,500 end-to-end and long-run thermal gates remain open. Additional pauses alone
cannot close that gap.

An isolated Unsloth2026.9.2 diagnostic using the same checkpoint, all28layers,
rank8 adapters, context512, batch1/accumulation4, fused FP32AdamW and exact CCE
measured1,687.60 compute tok/s,2.66GiB allocated VRAM and71C for four updates.
That is about10.1% faster than MOLT's1,532.16 compute result. This diagnostic has
no held-out evaluation and checks temperature only after an update, so it is not
a valid time-to-NLL or thermal-endurance victory for either engine. It does prove
that MOLT has not beaten the strongest available competitor compute path. The raw
artifact is `artifacts/mars-1p5b-20260904/competitor/unsloth-cce-exact-seed1337.json`.

## Why this could matter

With V=151936,B=4,L=512, oneFP32 logits tensor is B*L*V*4 =1.16GiB.
Checkpointing repeats forward work. Changing microbatch without changing
effectivebatch can amortize dispatch overhead, but can increase peakmemory.
An idealized frozen-base backward+forward workload of~4N FLOPs/token at1.5B
and2,500tok/s needs~15TFLOP/s, excluding attention, checkpoint recomputation,
dequantization, adapters and head details. This is a work estimate, not an
attainable-hardware-performance prediction.

## Preregistered screen

Same pretrained checkpoint, data/tokenizer, seed1337, context512, effectivebatch4,
AdamWlr2e-4, rank8 all-linearLoRA, 12updates/24,576tokens and evaluation every4:

- baseline: microbatch1xaccumulation4, checkpointing, noautocast;
- AMP: only enableautocast;
- AMPbatch4: only change geometry to4x1;
- AMPbatch4fused: only enablefused optimizer.

Local configurations are in `artifacts/mars-1p5b-20260904/`. Prepared data hashes
match the previous tokenization; source is TinyStories-valid with10%contiguous
tailholdout, not document-disjoint or a novel quality benchmark. Every case uses
the same processed tokens. Retain all failures/aborts and do not lowerquality or
raise thermal thresholds to pass. Timeincludes setup,evaluation,checkpointing;
boardenergy sampling excludes interpreterstartup/teardown. Short screens cannot
pass the sustainedspeed gate: a survivor needs >=500updates, independent sessions,
pairedseeds, targetNLL and controlled backgroundload, plus final-quarter speed.

Run after the model download completes:

```powershell
.\.venv\Scripts\python.exe -m molt_stream.cli --json train --config artifacts/mars-1p5b-20260904/baseline.json -y
.\.venv\Scripts\python.exe -m molt_stream.cli --json train --config artifacts/mars-1p5b-20260904/amp.json -y
.\.venv\Scripts\python.exe -m molt_stream.cli --json train --config artifacts/mars-1p5b-20260904/amp-batch4.json -y
.\.venv\Scripts\python.exe -m molt_stream.cli --json train --config artifacts/mars-1p5b-20260904/amp-batch4-fused.json -y
```

## Prior art and next hypothesis

- [PyTorch checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/): selective save/recompute is established.
- [Apple CCE](https://github.com/apple/ml-cross-entropy): outputhead tiling; its exact mode must be distinguished from gradient filtering.
- [Liger](https://github.com/linkedin/Liger-Kernel): fusedloss/kernel implementations, not a MOLT invention.
- [AQLoRA preprint](https://arxiv.org/abs/2608.23816): selected high-precision layer retention already studied; reported results are not independently reproduced here.
- [Zeus](https://www.usenix.org/system/files/nsdi23-you.pdf): energy/time-to-quality batch and power control prior art.
- [Predictive coding](https://arxiv.org/abs/2006.04182), [feedback methods](https://arxiv.org/abs/1904.05391), [LISA](https://arxiv.org/abs/2403.17919): brain-inspired/local/selective learning has prior art and changes optimization; not a drop-in proven speed fix.

Possible future MOLT experiment: jointly choose retained activations, cached
dequantized matrices and outputhead tiles under one measuredmemory budget.
ChooseitemsS maximizing sum(H*saved_time_i-switch_cost_i) with totalbytes under
headroom-reserve, then impose measuredenergy/quality/thermal constraints.
This resembles knapsack caching and model-predictive control; noveltyunproven.
Its unique value would need to come from evidence for a precise policy, not its
name or brain analogy. No competitor code copied. Model/code/dataset licenses
and patent questions must be reviewed separately before commercial distribution.
