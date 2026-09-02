# MOLT AI 1B — experimental model

MOLT AI 1B is a dense byte-level GPT-style research model trained locally with
the MOLT infrastructure. It is a systems demonstration, not a useful chatbot.

## Measured model

- Architecture: decoder-only causal Transformer
- Parameters: 1,008,742,400 unique trainable parameters
- Layers / width / heads: 20 / 2,048 / 16
- Context: 512 bytes
- Parameter and compute precision: BF16
- Optimizer: stateless SGD, learning rate 0.01
- Training: 1,024 updates, 524,288 bytes, seed 1337
- Data: first 17,500,000 bytes of `Dataset/TinyStories-valid.txt`
- Held-out validation: following 1,900,000 bytes
- Validation: 9.02877 to 4.15462 bits/byte
- End-to-end time: 302.229 seconds
- End-to-end throughput: 1,734.736 bytes/second
- Sampled GPU board energy: 20,840.277 J
- Checkpoint: 2,017,580,607 bytes
- Checkpoint SHA-256: `6999e705cf800f6a834c10327b476ac512a20dda7c50b3281751666a8b3fba85`

The validation decrease demonstrates learning on held-out bytes. Generated text
is still mostly gibberish: the run saw only 524,288 bytes, about 2.7% of the
available file and an extremely small pretraining budget for a random 1B model.
It did not correctly answer a question about Roxy and the icy hill.

## Generate text

From PowerShell:

```powershell
.\molt-ai.ps1 -Prompt "Once upon a time" -MaxNewBytes 120 -Temperature 0.8
```

The launcher uses the MOLT environment at `D:\Projects\MOLT`. It validates the
checkpoint hash before loading. Use `Temperature 0` for greedy output.

## Important paths

- Final model: `runs/20260901-023504-MOLT-AI-1B-TinyStories-v1-fa3fdd64/checkpoint.pt`
- Complete metrics: the run's `metrics.summary.json`
- Exact resolved configuration: the run's `spec.resolved.json`
- Training events: the run's append-only `events.jsonl`
- Generation tests: `generation-tests.json`
- Prepared-data hashes: `data/prepared/tinystories-v1/manifest.json`

Do not describe this checkpoint as ChatGPT-like, instruction-tuned, generally
useful, or fully trained. It is version `v1`, a reproducible local-training
mechanism and throughput artifact.
