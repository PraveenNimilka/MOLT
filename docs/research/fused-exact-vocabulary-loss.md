# MOLT exact frozen vocabulary loss

Status: component experiment passed; integrated promotion gate failed. The
analytical backend remains the default. No algorithmic novelty or end-to-end
performance victory is claimed.

## Scope and exact objective

Qwen2.5-1.5B has a 151,936-token vocabulary and a 1,536-wide hidden state. For
hidden row `h_i`, frozen classifier row `W_j`, and target `y_i`, MOLT computes:

```text
z_ij = h_i dot W_j
L_i  = log(sum_j exp(z_ij)) - z_i,y_i
dL_i/dh_i = sum_j softmax(z_i)_j W_j - W_y_i
```

The classifier must be frozen. The implementation changes storage and
scheduling only: it does not filter the vocabulary, alter targets, approximate
the gradient, or detach the decoder.

## Implemented operator

`molt::frozen_linear_cross_entropy` is registered through `torch.library`, with
fake/meta and autograd registrations. On supported CUDA inputs it processes a
bounded token chunk as follows:

1. cuBLAS projects the chunk through the frozen BF16/FP16 vocabulary head.
2. A Triton-Windows kernel performs online log-sum-exp, target extraction and
   cross-entropy reduction, then overwrites the private logits allocation with
   `(softmax - one_hot(target)) / token_count`.
3. cuBLAS multiplies that buffer by the frozen head to obtain the exact hidden
   gradient.

The implementation avoids the full-sequence FP32 logits tensor, but it **does
materialize one chunk-sized low-precision logits/gradient buffer**. It is not a
fully fused tiled GEMM-to-loss kernel and does not keep the entire operation in
SRAM. A true no-logits operator remains future work.

If Triton is unavailable or unsupported, `backend="auto"` uses the analytical
implementation. An explicit `backend="triton"` request fails rather than
silently changing the experiment. A trainable vocabulary head is rejected.

## Correctness verification

Tests cover CPU and CUDA `torch.library.opcheck`, BF16/FP32 parity against
unpartitioned cross-entropy, uneven vocabulary tails, non-contiguous and batched
inputs, repeated targets, extreme logits, invalid inputs, and refusal of a
trainable classifier. The complete repository suite is the release gate; run:

```powershell
uv run python -m pytest -q
```

## Component benchmark

The registered randomized 30-sample benchmark used 512 hidden rows, width
1,536, vocabulary 151,936, BF16 projection and FP32 reduction on the local RTX
4060 Laptop GPU:

| Exact implementation | Median latency |
|---|---:|
| MOLT autograd-partitioned | 30.04 ms |
| MOLT analytical-partitioned | 29.55 ms |
| Cut Cross-Entropy reference | 27.17 ms |
| MOLT Triton chunk operator | 18.60 ms |

The Triton component was 31.54% faster than CCE and passed the preregistered
15% component screen. With a 256-token chunk, measured temporary allocation was
about 158 MB; CCE used about 2.2 MB in the same scratch-memory probe. The raw
registered latency artifact is
`artifacts/mars-1p5b-20260904/exact-frozen-head-triton.json`.

## Integrated all-layer result

The decisive gate used Qwen2.5-1.5B with all 28 decoder layers active, rank-8
all-linear LoRA, context 512, batch 1, gradient accumulation 4, exact data order,
eight optimizer updates, resident activations, a cached frozen BF16 head and
four-window validation.

| Backend | Compute tok/s | End-to-end tok/s | Peak temperature | Result |
|---|---:|---:|---:|---|
| Analytical | 1,532.16 | 620.98 | 68 C | completed |
| Triton run 1 | 1,514.25 | 595.54 | 73 C | thermal abort |
| Triton run 2 | 1,509.48 | 520.63 | 73 C | thermal abort |

The Triton operator failed the required 10% integrated-update improvement and
also failed the registered 72 C thermal boundary in both measured runs. It is
therefore retained as an explicit experimental backend, not promoted as the
production default.

The analytical configuration crossed the 1,500 **update-compute** tok/s rung.
It did not cross 1,500 sustained end-to-end tok/s: thermal waiting and validation
reduced the measured result to 621 tok/s. A 16-update 64 C-target trial reached
1,556 compute tok/s and 759 end-to-end tok/s, but touched 72 C and was correctly
recorded as a thermal abort. These results are useful optimization evidence, not
completion of the sustained-performance milestone.

## What produced the integrated improvement

Profiling showed that repeatedly converting Qwen's 151,936 by 1,536 frozen FP32
head dominated more time than the loss reduction. MOLT now creates one
non-persistent BF16 compute cache for that frozen head while retaining the FP32
master for validation. Removing activation checkpoint recomputation also helped
and fits this 8 GB device at about 5.32 GiB peak PyTorch allocation. Decoder
GEMMs, NF4 dequantization, thermal waiting and evaluation are now larger
bottlenecks than the Triton loss kernel.

## Remaining falsifiable work

- Implement a genuinely fused tiled projection plus online reduction that does
  not materialize chunk logits, then repeat both memory and integrated gates.
- Profile a standards-compatible NF4-dequantization/LoRA fusion against
  bitsandbytes without changing quantization or optimizer semantics.
- Establish a reversible, verified power/clock operating point where hardware
  permits it. On this laptop, a 65 W limit request was not enforced and a clock
  lock was permission-denied; neither counts as a benchmark.
- Run three seeds and both run orders against a matched Unsloth environment,
  comparing time and board energy to identical held-out NLL.

## Prior art

- Cut Cross-Entropy: https://arxiv.org/abs/2411.09009
- Unsloth loss implementation: https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/loss_utils.py
- PyTorch custom operators and `opcheck`: https://docs.pytorch.org/docs/main/library.html
- PyTorch custom-operator autograd registration: https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html
