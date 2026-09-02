# Vision and scope

## Mission

Build a research-grade, local-first platform that makes efficiency claims about
AI training and adaptation falsifiable on consumer computers. The platform is
an instrument for discovering, rejecting, and reproducing methods—not evidence
that a new method already exists.

## Immediate scope

The first workload is a small autoregressive Transformer trained from scratch.
It is deliberately small enough to run on CPU and an 8 GiB GPU, yet large
enough to expose activation, optimizer-state, data-pipeline, and checkpointing
costs. Adaptation baselines (LoRA and QLoRA) come only after the measurement
pipeline has passed correctness and repeatability gates.

In scope:

- end-to-end time, quality, memory, GPU energy, utilization, storage, and recovery;
- deterministic small-model experiments on CPU and a single consumer GPU;
- isolated training-method, data-selection, and scheduling experiments;
- machine-readable configurations, artifacts, and comparisons;
- honest negative results and explicit uncertainty.

Out of scope until evidence justifies expansion:

- claims about frontier-scale training or vendor superiority;
- distributed training, NPU execution, custom CUDA kernels, and broad model serving;
- combining unvalidated optimizations;
- LLM-as-judge as the primary quality metric;
- estimated whole-system energy presented as measured energy.

## Research question and primary outcomes

At equal or better validation quality, can a local-first method improve one or
more of time-to-quality, GPU energy-to-quality, peak memory, reliability,
utilization, checkpoint/storage cost, or cost per successful run without an
unacceptable regression elsewhere?

Each experiment pre-registers a target quality and regression limits. Results
are reported for all runs, not only the best run. An optimization graduates only
after repeated trials and a controlled baseline comparison.

## Research integrity rules

1. A measured GPU-energy value covers the GPU board only; CPU, display, fans,
   storage, and power-supply losses require an external wall meter.
2. An out-of-memory avoidance result is not a speed or energy improvement.
3. Startup, compilation, warm-up, data loading, training, evaluation, and
   checkpointing are timed separately and together.
4. Baseline and candidate use identical data order, token budget, evaluation,
   and seeds unless the independent variable requires otherwise.
5. Failed and interrupted trials stay in the registry.
6. A method is not called novel until a dedicated prior-art and patent search is
   complete; this repository currently makes no novelty claim.
