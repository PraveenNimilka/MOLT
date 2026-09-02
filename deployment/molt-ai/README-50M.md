# MOLT AI 50M

MOLT AI 50M is a locally trained decoder-only Transformer with a proper
ByteLevel BPE tokenizer. It has a general TinyStories checkpoint and a separate
narrow question-answer specialist checkpoint.

## Model and training stack

- 49,883,520 unique trainable parameters
- nine Transformer blocks, width 640, ten attention heads, 512-token capacity
- tied token embedding/output weights and fused PyTorch causal SDPA
- 8,192-token Hugging Face ByteLevel BPE vocabulary
- FP32 master weights and FP32 AdamW moment state
- BF16 autocast matrix/attention computation
- no-decay AdamW group for biases and normalization parameters
- linear warmup followed by cosine learning-rate decay
- typed memory-mapped token streams and deterministic sequential epoch cursor
- atomic, hashed, rotating checkpoints

These are established mechanisms, not a novel-algorithm claim. Tokenizer choice
follows the Hugging Face Tokenizers BPE/ByteLevel APIs; mixed precision and
AdamW follow the official PyTorch APIs.

## Pretraining result

- Source: `Dataset/TinyStories-valid.txt`
- Train/validation: 4,413,694 / 231,510 BPE tokens
- Training budget: 8,835,072 tokens, just over two complete training epochs
- Time: 225.257 seconds
- End-to-end throughput: 39,222 tokens/s
- Validation perplexity: 9,954.04 to 33.70, monotonically decreasing
- Peak PyTorch allocation: 4.745 GB
- Peak device-wide VRAM: 6.565 GB
- Sampled GPU board energy: 16.544 kJ
- Base checkpoint SHA-256:
  `0a0462f5eb56629b4771c38c3a10c01c79a84c818e1b04e3c35e994b55bbb693`

The base model produces readable TinyStories-style paragraphs. Grammar,
consistency, and repetition are still weaker than a production language model.

## QA adaptation result

Prompt loss was masked so only answer tokens contributed to supervised loss.
The selected behavior checkpoint answered both the direct Roxy question and a
held-out paraphrase with “big leaves.” Its aggregate held-out QA loss was best
at step 20 and worsened later, so the selected step-200 QA checkpoint is a
narrow memorizing specialist, not a broad conversational model. The clean
early-stop control is retained separately.

QA checkpoint SHA-256:
`1e40797fd24a4ffbdc212849eaf9156f4b3718b95e039d40b8a8f324117ddb73`.

## Use

Story generation:

```powershell
.\molt-ai-50m.ps1 -Mode story -Prompt "Once upon a time"
```

Question answering:

```powershell
.\molt-ai-50m.ps1 -Mode qa -Prompt "What did Roxy put under her feet?"
```

The launcher selects the immutable base checkpoint for stories and the QA
specialist for registered question formatting. No online API or external model
is used.

## Primary references

- Hugging Face Tokenizers BPE quick tour:
  <https://huggingface.co/docs/tokenizers/python/latest/quicktour.html>
- PyTorch AdamW:
  <https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW>
- PyTorch automatic mixed precision:
  <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
- PyTorch scaled dot-product attention:
  <https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>
