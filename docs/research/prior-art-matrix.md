# Prior-art matrix — initial review

## Status and method

This is a scoped Milestone 0 review, not a novelty opinion or exhaustive patent
search. Sources are original papers, official documentation, dataset cards, and
official implementations where available. All benefits below are **reported by
the cited authors**, not reproduced by MOLT. “Fit” is our interpretation for the
audited Windows/8 GiB machine.

| Technique | Problem and mechanism | Reported benefit | Limitations / quality tradeoff | Hardware and implementation | Evidence / reproducibility | License / IP note | Fit |
|---|---|---|---|---|---|---|---|
| LoRA | Freeze base weights; train low-rank additive matrices | Far fewer trainable parameters and smaller task checkpoints; paper reports comparable adaptation quality on studied tasks | Primarily adaptation, not from-scratch training; rank and target modules matter | Standard PyTorch operations; no custom kernel required | Strong peer-reviewed evidence and official code | Official Microsoft implementation is MIT; patent search still pending | High for later adaptation baseline |
| QLoRA | Backpropagate into LoRA adapters through a frozen 4-bit NF4 base; double quantization and paged optimizer | Paper reports full-finetuning-quality adaptation at much lower memory on its tasks | Quantization kernels/platform support, task-dependent quality, not from-scratch training | Official code uses bitsandbytes; Windows support must be tested | Strong paper, code, many reported runs; local reproducibility unknown | Official QLoRA code says MIT; model and dataset licenses remain separate | Medium, after native Windows compatibility test |
| 8-bit optimizer states | Block-wise dynamic quantization of Adam statistics | Paper reports full-precision performance with a fraction of optimizer-state memory | Small tensors/outliers and kernel overhead can erase benefit; third-party kernel dependency | Usually bitsandbytes/CUDA | Peer-reviewed paper and open implementation; not reproduced here | Verify current package license and transitive binaries | Medium |
| 4-bit optimizer states | Quantize moments with smaller blocks and row/column outlier handling | Paper reports comparable accuracy across tested tasks with further memory savings | More numerical risk and implementation complexity; limited independent reproduction | Custom implementation likely required | NeurIPS 2023 paper and supplement; weaker ecosystem maturity | Patent and implementation-license review pending | Low initially |
| GaLore | Project gradients into periodically refreshed low-rank subspaces while updating full parameters | Paper reports up to 65.5% optimizer-state reduction; 8-bit variant reports larger savings | Projection/SVD overhead, hyperparameters, convergence and small-model relevance uncertain | PyTorch implementation available; no 24 GiB result transfers automatically to 8 GiB | Strong paper and code; limited independent local evidence | Official implementation license must be verified before reuse; patent search pending | High as isolated memory hypothesis |
| Activation checkpointing | Omit selected forward activations and recompute them during backward | Classic paper reports sublinear memory with added forward compute | Slower training; RNG/device movement can break equivalence if mishandled | Built into PyTorch; use non-reentrant API and explicit determinism tests | Established method with official docs | Framework license governs code use | Very high as control/baseline technique |
| FlashAttention / fused SDPA | Tile exact attention to reduce HBM traffic | Paper reports substantial end-to-end speedups, especially with longer sequences | Benefits depend on sequence shape/hardware; compile/startup costs; official external package is Linux/toolchain-oriented | PyTorch SDPA is baseline; external FlashAttention requires compatibility work | Strong papers/code, but external Windows fit is weak | Official repo permits use/modification; verify LICENSE at pin | Medium via built-in SDPA; low via external package |
| ZeRO-Offload | Move optimizer state/compute from GPU to CPU while minimizing transfers | Paper reports training much larger models on a single GPU | PCIe transfers and CPU/RAM pressure; this machine has only 16 GiB RAM | DeepSpeed-oriented, originally evaluated on data-center hardware | Strong systems paper and code; poor direct hardware match | Verify DeepSpeed and transitive licenses at pin | Low for first experiments |
| Progressive freezing | Freeze layers over training and remove their backward work | FreezeOut reports up to ~20% speedup in some CNNs, accuracy loss in some cases, and no gain in another architecture | Architecture-dependent; early decisions can irreversibly hurt optimization | Framework-native; easy to prototype | Older short paper, limited Transformer/local evidence | Algorithm patent search pending; prototype can be clean-room | High as cheap falsification target |
| Data deduplication | Remove exact/near duplicates and train-test overlap | Paper reports fewer steps to equal/better accuracy and much less memorized copying | Dedup compute, thresholds, false merges, and dataset-specific effect | Mostly CPU/storage preprocessing | Strong paper and released code; effect is corpus-dependent | Dataset licenses dominate; implementation license must be checked | Medium after baseline pipeline |
| Domain reweighting (DoReMi) | Train a proxy with group DRO to choose domain mixture weights | Paper reports reaching baseline accuracy in 2.6× fewer steps on one setup | Proxy cost is large locally; needs meaningful domains; scaling transfer uncertain | Standard accelerators but reported proxies are far larger than our baseline | Strong paper, not cheap to reproduce faithfully here | Dataset/model licenses and patent search pending | Low initially |
| Perplexity-based pruning | Use a smaller reference LM to score/select training data | Paper reports up to 1.45× fewer steps to commensurate performance | Scoring cost, bias, distribution shift, and leakage risk; gains below 2× gate | Can be staged on CPU/GPU; requires frozen scoring model | Recent paper with multi-scale experiments; local transfer unknown | Selected dataset and model licenses apply | High as a later data-efficiency test |
| TinyStories workload | Synthetic simple-language corpus supports useful evaluation of sub-10M models | Paper reports coherent language from very small models | Synthetic distribution; not representative of general LLM quality; judge-based evaluations are unsuitable as sole metric | Small enough for audited machine | Paper, public dataset, many derivatives | Dataset card states CDLA-Sharing-1.0 | High for instrumentation baseline only |
| Sequence packing | Pack variable-length samples to reduce padding while retaining attention/document boundaries | Krell et al. report up to 2x speedup and mathematical equivalence for their method | Incorrect masks can leak across documents; atom size/shuffling can alter quality; benefit vanishes for already-dense streams | Framework-native indexing/masking; fused-attention compatibility must be checked | Peer-reviewed/recent preprint evidence; not reproduced by MOLT | Reused implementations still need license/IP review | High after a variable-length baseline |
| Shape saturation/autotuning | Search context and micro-batch geometry that exposes enough parallel work to occupy the GPU under a memory constraint | PyTorch documents compiler/autotuning and CUDA-graph modes; no universal speed factor is promised | Raw tokens/s can improve while tokens-to-quality worsens; compilation, OOM, and changed optimization semantics are risks | Native PyTorch/CUDA; MOLT currently searches eager-mode shapes | Official framework guidance plus local measurements | MOLT code is original; the mechanism is established systems practice, not claimed novel | Very high as infrastructure |
| Zeus | Online exploration/exploitation over batch size and GPU power limit for recurring training jobs, with just-in-time energy profiling | Authors report 15.3–75.8% energy improvement on evaluated workloads | It explicitly navigates a time/energy tradeoff; exploration amortizes over recurring jobs; writable power limits are assumed | NVIDIA GPUs with energy/power telemetry and supported controls | Strong peer-reviewed systems work and released research project; not reproduced here | Algorithm is prior art; inspect implementation license before reuse | Closest controller baseline, but this laptop exposes no writable power limit |
| PowerTrain | Transfer-learned predictors estimate edge-training time and power across hardware power modes | Authors report accurate cross-device predictions and mode selection | Requires selectable CPU/GPU/memory frequency/core modes and representative profiling data | Evaluated on accelerated edge platforms such as Jetson | Primary preprint with implementation claims; not reproduced here | Prior art; implementation/license review pending | Useful comparison, but current laptop firmware does not expose the required modes |
| NVML clock/power event telemetry | Reports enforced power limit, clocks, temperature, utilization and reason bitmasks for power/thermal/other clock constraints | Measurement interface, not an optimization result | Sampling can miss transients; board energy excludes host CPU/display; support varies by device/driver | NVIDIA driver/NVML | Official NVIDIA documentation and local measurements | Vendor API license applies | Implemented as observability; no novelty claim |

## Primary sources

- LoRA: <https://arxiv.org/abs/2106.09685>; official implementation:
  <https://github.com/microsoft/LoRA>
- QLoRA: <https://arxiv.org/abs/2305.14314>; official implementation:
  <https://github.com/artidoro/qlora>
- 8-bit optimizers: <https://arxiv.org/abs/2110.02861>
- 4-bit optimizers: <https://openreview.net/forum?id=nN8TnHB5nw>
- GaLore: <https://arxiv.org/abs/2403.03507>
- Activation checkpointing: <https://arxiv.org/abs/1604.06174>; PyTorch API:
  <https://docs.pytorch.org/docs/stable/checkpoint>
- FlashAttention: <https://arxiv.org/abs/2205.14135>; official implementation:
  <https://github.com/Dao-AILab/flash-attention>
- ZeRO-Offload: <https://arxiv.org/abs/2101.06840>
- FreezeOut: <https://arxiv.org/abs/1706.04983>
- Dataset deduplication: <https://arxiv.org/abs/2107.06499>
- DoReMi: <https://arxiv.org/abs/2305.10429>
- Perplexity pruning: <https://arxiv.org/abs/2405.20541>
- TinyStories: <https://arxiv.org/abs/2305.07759>; dataset card:
  <https://huggingface.co/datasets/roneneldan/TinyStories>
- Sequence packing: <https://arxiv.org/abs/2107.02027>; document-preserving
  best-fit packing: <https://arxiv.org/abs/2404.10830>
- PyTorch performance tuning and compiler modes:
  <https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide>;
  pinned-memory/non-blocking transfers:
  <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>
- PyTorch scaled dot-product attention dispatch:
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention>
- Zeus: <https://arxiv.org/abs/2208.06102>
- PowerTrain: <https://arxiv.org/abs/2407.13944>
- NVIDIA NVML clock event reasons:
  <https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksEventReasons.html>

## Established facts, interpretations, and unknowns

**Established facts:** the audited GPU has 8 GiB VRAM; activation checkpointing
trades recomputation for stored activations; LoRA freezes the base and trains
low-rank updates; TinyStories is explicitly designed for small language models.

**Reported, not reproduced:** every numerical benefit in the matrix.

**Our interpretations:** a from-scratch small-model baseline exercises more of
the training system than an adaptation-only baseline; built-in PyTorch SDPA is a
fairer Windows baseline than an external kernel requiring a new toolchain;
GaLore, progressive freezing, and data pruning are plausible but must be tested
separately.

**Unknown:** whether any method improves time- or energy-to-quality on this
machine; whether Windows packages provide correct and competitive low-bit
kernels; whether results transfer to a sub-10M model; and whether any proposed
combination is novel or patent-clear.

## Prior-art work still required

Before claiming novelty for a specific method:

1. search Semantic Scholar/OpenAlex, arXiv, OpenReview, ACM/IEEE, and backward/
   forward citation graphs using the exact mechanism and synonyms;
2. reproduce the closest two implementable baselines;
3. search Google Patents, USPTO, EPO Espacenet, and WIPO PATENTSCOPE, recording
   queries, jurisdictions, families, claims of concern, and legal uncertainty;
4. inspect exact source, dataset, model-weight, and binary licenses at pinned
   revisions; and
5. have qualified counsel review material commercialization questions. This
   document is technical research, not legal advice.
