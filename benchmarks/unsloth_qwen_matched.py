"""Thermally matched all-layer Unsloth runner for MOLT comparison."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsloth-site", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--evaluation-interval", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--thermal-target-c", type=float, default=62.0)
    parser.add_argument("--thermal-guard-c", type=float, default=66.0)
    parser.add_argument("--thermal-abort-c", type=float, default=72.0)
    parser.add_argument("--thermal-power-target-watts", type=float, default=40.0)
    parser.add_argument("--thermal-initial-pause-seconds", type=float, default=0.30)
    parser.add_argument("--thermal-max-pause-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.steps, args.evaluation_interval, args.validation_batches) < 1:
        parser.error("step and validation counts must be positive")
    if not args.thermal_target_c <= args.thermal_guard_c < args.thermal_abort_c:
        parser.error("thermal boundaries must satisfy target <= guard < abort")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    sys.path.insert(0, str(Path(args.unsloth_site).resolve()))
    from cut_cross_entropy import linear_cross_entropy
    from unsloth import FastLanguageModel
    from molt_stream.core.specs import DataSpec
    from molt_stream.data.bytes import MMapTokenBatcher
    from molt_stream.measurement.telemetry import NVMLTelemetry, integrate_board_energy
    from molt_stream.measurement.thermal import (
        ThermalCruiseController,
        cooling_pause,
        latest_telemetry_point,
        wait_for_thermal_headroom,
    )
    from molt_stream.training.evaluation_guard import (
        EvaluationThermalStop,
        guarded_decoder_evaluation,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    telemetry = NVMLTelemetry(0.05)
    telemetry.start()
    started = time.perf_counter()
    model, _ = FastLanguageModel.from_pretrained(
        model_name=str(Path(args.model).resolve()),
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        fix_tokenizer=False,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("Matched benchmark requires FP32 adapter masters")
    optimizer = torch.optim.AdamW(parameters, lr=0.0002, fused=True)
    train = MMapTokenBatcher(
        DataSpec(
            str(Path(args.train_data).resolve()), context_length=512,
            storage_dtype="int32", packing="contiguous", sequential=True,
        ),
        seed=args.seed,
        device="cuda",
    )
    validation = MMapTokenBatcher(
        DataSpec(
            str(Path(args.validation_data).resolve()), context_length=512,
            storage_dtype="int32", packing="contiguous", sequential=True,
        ),
        seed=args.seed + 1,
        device="cuda",
    )
    controller = ThermalCruiseController(
        target_c=args.thermal_target_c,
        abort_c=args.thermal_abort_c,
        lookahead_seconds=2.5,
        stability_band_c=1.5,
        initial_pause_seconds=args.thermal_initial_pause_seconds,
        maximum_pause_seconds=args.thermal_max_pause_seconds,
        power_target_watts=args.thermal_power_target_watts,
    )
    thermal_pause_seconds = 0.0
    thermal_stop_reason: str | None = None

    def gate() -> bool:
        nonlocal thermal_pause_seconds, thermal_stop_reason
        decision = wait_for_thermal_headroom(
            telemetry.thermal_point,
            target_c=args.thermal_guard_c,
            abort_c=args.thermal_abort_c,
        )
        thermal_pause_seconds += decision.pause_seconds
        thermal_stop_reason = decision.stop_reason
        return decision.stop_reason is None

    def exact_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        base = model.get_base_model()
        hidden = base.model(input_ids=x, use_cache=False, return_dict=True).last_hidden_state
        return linear_cross_entropy(
            hidden,
            base.get_output_embeddings().weight,
            y,
            shift=False,
            reduction="mean",
            filter_eps=None,
        )

    def current_temperature() -> float | None:
        point = telemetry.thermal_point()
        return None if point is None else point.gpu_temperature_c

    @torch.no_grad()
    def validation_nll() -> float | None:
        state = validation.state_dict()
        model.eval()
        losses: list[float] = []
        try:
            for _ in range(args.validation_batches):
                if not gate():
                    return None
                x, y = validation.batch(1)
                with guarded_decoder_evaluation(model, gate):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        losses.append(float(exact_loss(x, y)))
                if not gate():
                    return None
        except EvaluationThermalStop:
            return None
        finally:
            validation.load_state_dict(state)
            model.train()
        return sum(losses) / len(losses)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial = validation_nll()
    evaluations: list[dict[str, float | int]] = []
    if initial is not None:
        evaluations.append({
            "step": 0,
            "nll": initial,
            "perplexity": math.exp(initial),
            "end_to_end_seconds": time.perf_counter() - started,
            "board_energy_joules": integrate_board_energy(telemetry.points) or 0.0,
        })
    compute_seconds = 0.0
    tokens = 0
    step = 0
    state = "completed" if initial is not None else "thermal_abort"
    while step < args.steps and state == "completed":
        optimizer.zero_grad(set_to_none=True)
        update_started = time.perf_counter()
        pause_before_update = thermal_pause_seconds
        for _ in range(4):
            if not gate():
                state = "thermal_abort"
                break
            x, y = train.batch(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = exact_loss(x, y) / 4.0
            torch.cuda.synchronize()
            if not gate():
                state = "thermal_abort"
                break
            loss.backward()
            torch.cuda.synchronize()
            if not gate():
                state = "thermal_abort"
                break
        if state != "completed":
            break
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - update_started
        compute_seconds += max(
            0.0, elapsed - (thermal_pause_seconds - pause_before_update)
        )
        step += 1
        tokens += 2048
        decision = controller.update(
            latest_telemetry_point(telemetry.points), step_seconds=max(elapsed, 1e-6)
        )
        if decision.abort:
            state = "thermal_abort"
            thermal_stop_reason = "thermal abort boundary reached"
            break
        if decision.pause_seconds:
            thermal_pause_seconds += cooling_pause(
                decision.pause_seconds,
                current_temperature,
                recovery_c=args.thermal_target_c,
                abort_c=args.thermal_abort_c,
            )
        if step == args.steps or step % args.evaluation_interval == 0:
            nll = validation_nll()
            if nll is None:
                state = "thermal_abort"
                break
            evaluations.append({
                "step": step,
                "nll": nll,
                "perplexity": math.exp(nll),
                "end_to_end_seconds": time.perf_counter() - started,
                "board_energy_joules": integrate_board_energy(telemetry.points) or 0.0,
            })
    measured = telemetry.stop()
    seconds = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "competitor": "unsloth",
        "unsloth_version": "2026.9.2",
        "unsloth_zoo_version": "2026.9.1",
        "state": state,
        "thermal_stop_reason": thermal_stop_reason,
        "model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "context_length": 512,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "steps": step,
        "tokens": tokens,
        "evaluations": evaluations,
        "seconds": seconds,
        "compute_seconds": compute_seconds,
        "compute_tokens_per_second": tokens / compute_seconds if compute_seconds else None,
        "end_to_end_tokens_per_second": tokens / seconds if seconds else None,
        "thermal_pause_seconds": thermal_pause_seconds,
        "thermal_policy": {
            "target_c": args.thermal_target_c,
            "guard_c": args.thermal_guard_c,
            "abort_c": args.thermal_abort_c,
            "average_power_target_watts": args.thermal_power_target_watts,
            "initial_pause_seconds": args.thermal_initial_pause_seconds,
            "maximum_pause_seconds": args.thermal_max_pause_seconds,
        },
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "telemetry": {key: value for key, value in measured.items() if key != "points"},
        "semantic_note": "Already-shifted targets; exact unfiltered CCE; 2048 predicted tokens/update.",
    }
    (output / "metrics.summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output / "telemetry.samples.json").write_text(
        json.dumps({"points": measured["points"], "errors": measured["errors"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if state == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
