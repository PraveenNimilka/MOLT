# All-layer Qwen2.5-1.5B memory and thermal frontier (2026-09-06)

## Registered question

Can MOLT train all 28 Qwen2.5-1.5B decoder layers at context 512 with the
same 2,048-token AdamW update, held-out objective and data order while staying
below 3 GiB allocated CUDA memory and sustaining at least 1,200 end-to-end
tokens/second below the 72 C safety boundary?

All results below use B1/G4, rank-8 all-linear QLoRA, BF16 tensor-core compute,
FP32 AdamW state, exact partitioned cross-entropy, seed 1337 and four held-out
validation windows unless explicitly noted. Top-layer-only experiments are
excluded.

## Mechanisms tested

1. Full activation checkpointing: established recomputation baseline.
2. Pinned-CPU saved-tensor offload: exact gradients; all layers active.
3. Packed symmetric INT4 saved activations: experimental approximate gradient
   storage. This is related to established activation-compression research and
   is not claimed as novel.
4. Restore the frozen tied input/output embedding to the checkpoint's original
   BF16 storage after k-bit preparation.
5. Frozen RMSNorm with FP32 reduction and weight arithmetic but a BF16 residual
   stream and analytical hidden gradient.
6. Sparse layer checkpoint placement plus exact vocabulary-loss chunk tuning.

PyTorch describes activation checkpointing as a memory/recomputation trade-off
and exposes selective checkpoint policies; MOLT's layer placement is an
application of that established mechanism:
<https://pytorch.org/blog/activation-checkpointing-techniques/>.
Hugging Face similarly documents the expected checkpointing slowdown:
<https://huggingface.co/docs/transformers/grad_checkpointing>.

## Measured results

| Arm | Steps | Compute tok/s | End-to-end tok/s | Peak allocated GiB | Peak board-used GiB | Peak C | Final NLL | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Resident analytical reference | 8 | 1,532 | 621 | 5.318 | 6.02 | 68 | 1.53397 | Speed reference; memory fail |
| Full checkpoint reference | 8 | 975 | 324 | 2.72 | 3.39 | 77 | recorded artifact | Memory pass; speed/thermal fail |
| CPU saved-activation offload | 8 | 502 | 413 | 2.635 | 3.314 | 71 | 1.53397 | Reject: PCIe/RAM cost |
| B2/G2 full checkpoint | 8 | 938 | 422 | 2.805 | 3.461 | 80 | 1.53457 | Reject: slower and unsafe |
| INT4 saves, BF16+FP32, chunk 128 | 8 | 1,136 | 569 | 3.186 | 4.022 | 72 | 1.53467 | Reject: misses memory and thermal gates |
| BF16 tied head + INT4 large saves + BF16 RMSNorm | 4 | 1,313 | 386 | 3.029 | 3.799 | 72 | 1.56241 | Reject: misses both gates |
| BF16 tied head + BF16 RMSNorm, no checkpoint | 4 | 1,805 | 598 | 3.263 | 3.930 | 70 | 1.56133 | Strong compute result; memory/end-to-end fail |
| Exact BF16 RMSNorm + stride 8 + loss chunk 96 | 4 | 1,727 | 592 | 2.956 | 3.576 | 71 | 1.56058 | Short memory/compute pass; end-to-end fail |
| Same exact arm, endurance | 15/16 | 1,084 | 401 | 2.956 | 3.621 | 73 | 1.53307 | Reject for production: ten recoveries, thermal stop |
| Stride 8 + separated 62/66/72 C control | 8 | 1,636 | 624 | 2.956 | 3.597 | 70 | 1.53307 | Stable short reference; no recoveries, but end-to-end gate fails |
| Full checkpoint + BF16 RMS | 8 | 1,194 | 464 | 1.944 | 2.803 | 72 | 1.53307 | Board-memory gate passes; delayed thermal boundary fails |
| Full checkpoint + BF16 adapter shadows | 7/8 | 1,189 | 392 | 1.944 | 2.742 | 73 | no final evaluation | Reject: three recoveries and thermal stop |
| Stride 8 + fused Triton loss | 5/8 | 1,617 | 414 | 2.817 | 3.396 | 74 | no final evaluation | Reject: slower, incomplete and thermally invalid |

Artifact roots:

- `artifacts/mars-1p5b-20260904/activation-offload/`
- `artifacts/mars-1p5b-20260904/b2g2-checkpoint/`
- `artifacts/mars-1p5b-20260904/int4-activations/`
- `artifacts/mars-1p5b-20260904/bf16-residual/`
- `artifacts/mars-1p5b-20260904/bf16-rms-stride8/`

## Conclusions

- The original 445 MiB cache report understated the tied-weight issue. K-bit
  preparation stores the checkpoint's BF16 tied embedding/head as FP32, then
  the optimized loss adds a BF16 copy. Restoring the frozen tied tensor to its
  source BF16 dtype lets the loss cache alias it and materially reduces memory.
- Standard Qwen2 RMSNorm with an FP32 frozen weight promotes the residual output
  to FP32. The BF16-output analytical operator produced the largest compute and
  memory-traffic improvement, while retaining FP32 RMS reduction.
- The claim that tiny dataset H2D copies explain the pipeline drain is false for
  this workload. Each microbatch transfers only token IDs; measured thermal
  waiting dominates the compute-to-end-to-end gap.
- The requested joint gate was **not passed**. The best exact short arm passes
  allocated memory and raw compute, but the laptop cannot dissipate sustained
  heat at that duty cycle under the 72 C boundary. Its long run is the governing
  result.
- A lower GPU voltage/clock operating point or stronger cooling is required to
  test whether the physical duty-cycle limit can move. This laptop exposes no
  NVML fan control, and prior non-administrative power/clock requests were not
  applied. Removing thermal guards would invalidate the comparison.
- The 65 W power-limit capability check on 2026-09-06 was rejected by NVML with
  `Insufficient Permissions`; the original 140 W enforced limit remained in
  place and was verified. Software pacing is therefore the only controllable
  power mechanism in this non-administrative session.
- A distinct microbatch guard is now configurable above the cruise target. It
  removes the previous hidden requirement to cool below the cruise target
  before every forward and backward segment. A 50 W point still overshot; the
  conservative 40 W point completed without recovery, demonstrating correct
  control separation but not the requested throughput.
- The matched Unsloth kill runner uses the same token stream, all-layer adapter
  inventory, FP32 fused AdamW, exact unfiltered loss, validation windows and
  thermal boundary. Two cold screens stopped Unsloth after 3-4 updates at 72 C;
  a 30 W pacing arm reached six updates and then peaked at 73 C. These are short
  falsification results, not a three-seed or endurance victory claim.

## Mid-update software fallback

MOLT now has an opt-in Qwen2 decoder-boundary guard using
`qlora_intra_step_boundary_layer` and `qlora_intra_step_pause_ms`. It synchronizes
and samples thermals at an exact forward/backward graph boundary; it does not
alter tensors, gradients, layer count, optimizer semantics, or data order. CPU
tests confirm bit-identical loss and gradients and unconditional hook cleanup.

The registered 14/28 boundary with a 10 ms dwell did **not** improve this 1.54B
workload. Compute throughput fell from 1,710 to 1,566 tok/s and end-to-end
throughput from 613 to 492 tok/s, while both screens reached 72 C. The candidate
preserved the identical final four-window NLL of 1.533068. This rejects a fixed
midpoint dwell as an acceleration method; it remains an experimental safety
fallback only.

An elevated 65 W / 1,800-1,950 MHz operating-point test remains pending. The
2026-09-06 shell was non-administrative, and NVIDIA correctly denied the clock
request. No clock or power setting was changed.

### Graphics-clock efficiency screen

An Administrator session confirmed that this laptop refuses writable power
limits but accepts a temporary 1,800-1,950 MHz graphics-clock lock. The lock was
reset in a PowerShell `finally` block and post-run idle telemetry confirmed the
reset.

The all-layer Qwen2.5-1.5B B1/G4 context-512 screen completed 8/8 updates at
1,588.5 committed-compute tok/s and 1,485.6 training-loop tok/s. Full end-to-end
throughput, including setup and validation, was 1,060.2 tok/s. Peak temperature
was 63 C, board energy was 703.3 J (0.04293 J/token), and there were no thermal
pauses, recoveries, discarded tokens, post-run violations, or measured thermal
clock events. NLL changed from 1.663530 to 1.533068.

The identical immediately-following unlocked control was heat-soak confounded,
but it is still an operational falsification: clocks rose to 2,370-2,520 MHz,
temperature reached 72 C, 5,632 uncommitted tokens were rolled back, and the run
stopped after 2/8 updates. It must not be used as a statistical speed ratio.

This passes the short safety/feasibility screen, not the endurance gate. The
locked run's late/early rate ratio was 0.795. A thermally normalized AB/BA run of
at least 30 minutes remains required before promoting a clock range or claiming
1,200 sustained end-to-end tok/s.

## Promotion status

None of the new profiles is a production default. The BF16 tied-weight and
frozen-RMSNorm policies remain explicit until multiple seeds, resume parity and
long-duration validation pass. Packed activation storage remains experimental.
