"""Legacy Unsloth Qwen diagnostic, not an accepted matched benchmark.

This short probe has no validation phase and only checks thermal state after
updates. Do not compare its compute rate with MOLT's thermally guarded sessions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsloth-site", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--last-layers", type=int)
    parser.add_argument("--loss-backend", choices=("full-logits", "cce-exact"),
                        default="cce-exact")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.steps < 1:
        parser.error("--steps must be positive")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    sys.path.insert(0, str(Path(args.unsloth_site).resolve()))
    from unsloth import FastLanguageModel
    from molt_stream.core.specs import DataSpec
    from molt_stream.data.bytes import MMapTokenBatcher
    from molt_stream.measurement.telemetry import NVMLTelemetry

    seed = 1337
    random.seed(seed)
    torch.manual_seed(seed)
    telemetry = NVMLTelemetry(0.05)
    telemetry.start()
    started = time.perf_counter()
    model, _ = FastLanguageModel.from_pretrained(
        model_name=str(Path(args.model).resolve()), max_seq_length=512,
        dtype=torch.bfloat16, load_in_4bit=True, fix_tokenizer=False,
        use_gradient_checkpointing="unsloth", random_state=seed,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=seed,
        finetune_last_n_layers=args.last_layers,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("Matched optimizer requires FP32 adapter parameters")
    optimizer = torch.optim.AdamW(parameters, lr=0.0002, fused=True)
    batcher = MMapTokenBatcher(
        DataSpec(str(Path(args.data).resolve()), context_length=512,
                 storage_dtype="int32", packing="contiguous", sequential=True),
        seed=seed, device="cuda",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    setup_seconds = time.perf_counter() - started
    compute_seconds = 0.0
    tokens = 0
    losses: list[float] = []
    state = "completed"
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        update_started = time.perf_counter()
        loss_total = 0.0
        for _ in range(4):
            x, y = batcher.batch(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.loss_backend == "cce-exact":
                    from cut_cross_entropy import linear_cross_entropy
                    base = model.get_base_model()
                    hidden = base.model(
                        input_ids=x, use_cache=False, return_dict=True
                    ).last_hidden_state
                    loss = linear_cross_entropy(
                        hidden, base.get_output_embeddings().weight, y,
                        shift=False, reduction="mean", filter_eps=None,
                    ) / 4
                else:
                    logits = model(input_ids=x, use_cache=False).logits
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
                    ) / 4
            loss.backward()
            loss_total += float(loss.detach())
        optimizer.step()
        torch.cuda.synchronize()
        compute_seconds += time.perf_counter() - update_started
        tokens += 2048
        losses.append(loss_total)
        point = telemetry.thermal_point()
        if point is None or point.gpu_temperature_c is None or point.gpu_temperature_c >= 72:
            state = "thermal_stop"
            break
    measured = telemetry.stop()
    result = {
        "competitor": "unsloth", "unsloth_version": "2026.9.2",
        "unsloth_zoo_version": "2026.9.1", "state": state,
        "model": str(Path(args.model).resolve()), "seed": seed,
        "context_length": 512, "batch_size": 1, "gradient_accumulation": 4,
        "last_layers": args.last_layers, "loss_backend": args.loss_backend,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "steps": len(losses), "tokens": tokens, "losses": losses,
        "setup_seconds": setup_seconds, "compute_seconds": compute_seconds,
        "compute_tokens_per_second": tokens / compute_seconds if compute_seconds else 0.0,
        "end_to_end_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "board_energy_joules": measured.get("gpu_board_energy_joules"),
        "peak_gpu_temperature_c": measured.get("peak_gpu_temperature_c"),
        "semantic_note": "Already-shifted targets; 2048 predicted tokens/update; CCE disables filtering.",
        "limitations": ["No held-out evaluation or shared thermal protocol",
                        "Four default updates are not a sustained-throughput measurement",
                        "Requested last-layer count is not an adapter-inventory parity test"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if state == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
