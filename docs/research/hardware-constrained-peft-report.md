# Hardware-Constrained PEFT on a Consumer Laptop

## Design, verified single-run endurance, and a preregistered iso-clock protocol

**Status:** research report for MOLT 0.11 alpha. The endurance result is verified
from local raw artifacts. The MOLT-versus-Unsloth comparison remains unverified
and is not a release claim.

## Abstract

Fine-tuning on a mobile discrete GPU is constrained by both accelerator memory
and the cooling capacity of the complete laptop. A configuration that maximizes
short-burst kernel throughput may lose wall-clock throughput after the chassis
heat-soaks. Windows adds a second constraint: under WDDM, process residency is
budgeted and may change under memory pressure rather than equaling nominal VRAM.

MOLT is a Windows-first QLoRA runtime that treats memory allocation, board
telemetry, thermal boundaries, checkpoint recovery, and optimizer semantics as
one measured system. On one RTX 4060 Laptop GPU, one seed of an all-28-layer
Qwen2.5-1.5B adaptation completed 2,400 updates and 4,915,200 predicted tokens
in 3,725.65 seconds. It measured 1,319.29 end-to-end tokens/s, 0.03625 board
joules/token, a 71 C peak, and a 0.973 last-quarter/first-quarter throughput
ratio. Held-out NLL changed from 1.66353 to 1.10445. This is evidence of
single-machine endurance, not evidence that MOLT is generally faster than
another runtime.

We specify a matched iso-clock evaluation against Unsloth 2026.9.2. The
repository does not currently contain the claimed three-seed aggregate needed
to complete that comparison. Consequently, competitor deltas supplied outside
the artifact store are rejected from headline results.

## 1. Problem formulation

Let a training run produce a sequence of validation observations

\[
  E = \{(s_i, t_i, J_i, L_i)\}_{i=0}^{n},
\]

where \(s_i\) is the committed optimizer update, \(t_i\) is elapsed wall time,
\(J_i\) is integrated GPU board energy, and \(L_i\) is held-out negative
log-likelihood. For a preregistered target \(L^*\), the primary system metrics
are interpolated time-to-loss \(t(L^*)\) and energy-to-loss \(J(L^*)\), not the
fastest individual update.

The experiment is valid only when both arms preserve the same model revision,
token order, effective batch, trainable parameter inventory, optimizer
semantics, validation objective, thermal ceiling, clock range, and starting
thermal condition. Uncommitted work rolled back by a thermal tripwire is not
counted as learned tokens.

WDDM 2.x manages discrete-GPU memory through segments and process residency
budgets. Therefore MOLT records both PyTorch peak allocation and NVML board-used
memory; neither is presented as a substitute for the other.

## 2. System architecture

### 2.1 Quantized base and trainable state

The pretrained base uses bitsandbytes NF4 with double quantization. The model
base is frozen and gradients update rank-8 LoRA adapters in all seven standard
linear projection classes across all 28 decoder layers. The adapters, their
gradients, and AdamW master moments remain unquantized FP32. The effective update
contains 2,048 predicted tokens (context 512, batch one, accumulation four).

### 2.2 Tied vocabulary head and exact chunked loss

For Qwen2.5-1.5B, the tied embedding/output weight is restored to its checkpoint
BF16 storage rather than retaining an unnecessary FP32 copy. Cross-entropy is
computed in vocabulary chunks of 96 with an analytical hidden gradient. The
method avoids retaining the complete token-by-vocabulary logit tensor while
preserving the exact, unfiltered cross-entropy objective. If the output weight
requires a gradient, MOLT dispatches to the dual-gradient path instead.

### 2.3 Selective activation checkpoint placement

MOLT's stride-8 profile checkpoints selected layer boundaries instead of every
decoder layer. This is an application of established activation checkpointing,
not a newly invented algorithm. It trades saved activations for recomputation
at fewer boundaries and must be assessed against a matched quality objective.

### 2.4 Telemetry and recovery

NVML is sampled every 100 ms for board power, temperature, utilization, clocks,
clock-event reasons, and board-used memory. Board energy is the trapezoidal
integral of sampled board power. It excludes CPU, display, SSD, fans, and power
conversion losses. Thermal decisions occur inside accumulated updates. A stop
before `optimizer.step()` restores the token cursor and RNG state, discards
partial gradients, and checkpoints only committed work through temporary-write,
flush, digest, and atomic replacement.

### 2.5 Clock control boundary

The endurance profile requests 1,500--1,650 MHz through NVIDIA's supported
graphics-clock interface. This changes a machine-wide GPU setting, requires
explicit authorization and administrator privileges, and is always paired with
an automatic-clock reset in `finally`. Telemetry, rather than command success
alone, verifies the operating range.

## 3. Verified endurance result

The governing artifact is locally retained at
`artifacts/mars-1p5b-20260904/endurance-1hour-1800-1950/20260906-123338-qlora-4a30a884/`.
The large raw artifact is intentionally excluded from source distributions.

| Metric | Measured value |
| --- | ---: |
| Model / active layers | Qwen2.5-1.5B / 28 of 28 |
| Trainable LoRA parameters | 9,232,384 |
| Updates / predicted tokens | 2,400 / 4,915,200 |
| Total elapsed time | 3,725.652 s |
| End-to-end throughput | 1,319.286 tokens/s |
| Training-loop throughput | 1,320.642 tokens/s |
| Committed-update compute throughput | 1,399.520 tokens/s |
| Board energy | 178,164.504 J |
| Board energy per token | 0.036248 J/token |
| Mean board power | 47.847 W |
| Peak GPU temperature | 71 C |
| First-quarter throughput | 1,347.116 tokens/s |
| Last-quarter throughput | 1,310.989 tokens/s |
| Last/first throughput ratio | 0.97318 |
| PyTorch peak allocation | 2.956 GiB |
| NVML peak board-used memory | 3.750 GiB |
| Recorded thermal pacing | 190.093 s |
| Recovery cycles / discarded tokens | 0 / 0 |
| Initial / final held-out NLL | 1.663530 / 1.104453 |

Validation NLL is not monotonic at every 100-step observation, but its final
value is materially below its initial value. No claim of monotonic validation
loss is made.

## 4. Factorial ablation status

The proposed 2x2 factors are clock constraint (unconstrained versus
1,500--1,650 MHz) and the bundled MOLT memory execution profile (off versus on).
The repository currently contains useful short arms, but not a complete,
thermally normalized 2x2 experiment with three seeds and both run orders.

| Condition | Repository evidence | Publication decision |
| --- | --- | --- |
| Uncapped, reference memory path | Short runs often reached the 72 C boundary | Retain as negative operational evidence |
| Capped, reference memory path | No complete matched three-seed factorial artifact | Pending |
| Uncapped, MOLT memory path | Short and heat-soak-confounded failures exist | Do not derive a speed ratio |
| Capped, MOLT memory path | One verified hour-long MOLT run | Endurance evidence only |

The externally supplied figures `468.52`, `801.10`, and `1,214.50` tokens/s do
not appear in the checked artifact registry as a complete factorial result.
They are therefore not treated as verified measurements in this report.

## 5. Iso-clock comparison protocol and status

The preregistered comparison uses Qwen2.5-1.5B, all 28 decoder layers,
9,232,384 active rank-8 LoRA parameters, context 512, B1/G4, FP32 fused AdamW,
the same int32 token stream, four held-out validation windows, and a 72 C hard
boundary. Both engines run at 1,500--1,650 MHz. Seeds are 1337, 2027, and 4099
in alternating AB/BA order after an identical cold-start criterion.

The one-click screen is:

```powershell
python benchmarks/reproduce_isoclock.py
```

It runs one seed for 16 updates and is a protocol smoke test. The separate
`benchmarks/qwen_unsloth_ab.py` harness is the multi-seed acceptance runner.

| Required evidence | Current state |
| --- | --- |
| Three independent seeds | Not present |
| Alternating run order | Harness implemented; complete result absent |
| Both engines complete below 72 C | Not demonstrated across three pairs |
| Same initial and final held-out objective within 1% | Enforced by new screen; aggregate absent |
| Time and energy to the same NLL | Harness implemented; aggregate absent |
| At least 30 minutes per arm | Not demonstrated |

The supplied aggregate of 1,214.50 versus 886.22 tokens/s and its derived 37.0%
advantage cannot be independently reconstructed from repository artifacts.
Publishing it as verified would violate MOLT's comparison methodology.

## 6. Timing ledger

The one-hour artifact records 1.775 s of setup, 3,721.826 s in the training-loop
scope, 20.516 s of validation work, and 0.319 s for the final checkpoint. These
scopes are not all additive: validation is invoked from the training loop.
Accordingly, MOLT reports the measured total of 3,725.652 s and does not sum
overlapping subscopes. The supplied 27.01 s MOLT and 37.62 s Unsloth ledgers are
not present as reconstructable raw artifacts and remain unverified.

## 7. Causal interpretation

The verified result supports a narrower chain:

1. NF4 base storage, BF16 tied-head storage, exact chunked vocabulary loss, and
   selective checkpoint placement keep PyTorch allocation below 3 GiB for this
   workload.
2. The temporary clock range lowers short-burst power relative to unrestricted
   boost behavior on this laptop, allowing a completed hour-long run below the
   72 C boundary.
3. Throughput remained stable within 2.7% from the first to last quarter while
   the quality objective improved.

It does not establish that memory compaction alone causes a 51.6% speedup, that
MOLT eliminates thermal pacing, or that MOLT beats Unsloth. Those questions
require the incomplete factorial and multi-seed experiments.

## 8. Limitations and applicability

- The endurance result covers one laptop, one GPU, one seed, one model, and one
  dataset stream.
- The GPU clock command changes the whole selected device while active and may
  be unsupported by another laptop firmware or driver.
- Board energy is not wall-plug system energy.
- The runtime depends on third-party quantization and model libraries whose
  versions and licenses remain independent.
- WDDM budgets vary with concurrent desktop workloads; PyTorch allocation is
  not total board residency.
- Selective checkpointing can change floating-point execution order and must be
  revalidated for each model family and precision path.
- Sixteen updates are appropriate for a kill test, not a generalization or
  convergence claim.

## 9. Reproducibility and evidence gate

Before any competitor headline is published, retain for every arm: resolved
configuration, model/dataset hashes and licenses, stdout/stderr, 100 ms telemetry,
evaluation crossings, package lock, GPU/driver identity, start temperature,
clock verification, run order, and commit SHA. Aggregate all three seeds rather
than selecting the best run. An invalid or aborted arm remains a result and is
not silently rerun under a different protocol.

## References

1. Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023.
2. Microsoft, [Process Residency Budgets in WDDM 2](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/process-residency-budgets).
3. Microsoft, [GPU Segments](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-segments).
4. NVIDIA, [`nvidia-smi` GPU clock controls](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
5. PyTorch, [Activation Checkpointing Techniques](https://pytorch.org/blog/activation-checkpointing-techniques/).
