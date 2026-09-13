"""Differential full-model NF4 training gate; not a throughput benchmark."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from benchmarks.static_qlora_update_gate import _loss_builder, _validation_nll
from molt_stream.core.specs import load_spec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.methods.nf4_lora import enable_scheduled_nf4_lora
from molt_stream.training.qlora import _build_qlora_model, _build_qlora_optimizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--reference-control", action="store_true")
    parser.add_argument("--analytical-math", action="store_true")
    parser.add_argument("--suffixes", default="", help="Comma-separated candidate projections")
    parser.add_argument("--loss-backend", choices=("auto", "analytical", "triton"))
    parser.add_argument("--sdpa", choices=("auto", "math", "efficient", "cudnn"))
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--split-loss", action="store_true", help="Experimental loss on candidate only")
    parser.add_argument("--cache-backward-weights", action="store_true")
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=8)
    parser.add_argument("--cool-start-c", type=int, default=70)
    parser.add_argument("--cool-resume-c", type=int, default=60)
    args = parser.parse_args()
    if min(
        args.steps,
        args.validation_batches,
        args.checkpoint_interval,
        args.progress_interval,
    ) < 1:
        parser.error("steps, validation batches, and intervals must be positive")
    if not 0 < args.cool_resume_c < args.cool_start_c < 84:
        parser.error("cooling must satisfy 0 < resume < start < 84 C")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = replace(load_spec(args.config), seed=args.seed)
    if args.analytical_math:
        spec = replace(spec, qlora_loss_backend="analytical", qlora_sdpa_kernel="math")
    if args.loss_backend:
        spec = replace(spec, qlora_loss_backend=args.loss_backend)
    if args.sdpa:
        spec = replace(spec, qlora_sdpa_kernel=args.sdpa)
    if spec.gradient_accumulation != 1:
        raise ValueError("This gate currently requires gradient_accumulation=1")
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    peak_c = 0

    def cool() -> None:
        nonlocal peak_c
        started = time.monotonic()
        temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
        peak_c = max(peak_c, temperature)
        if temperature >= 84:
            raise RuntimeError("Thermal abort at 84 C")
        if temperature < args.cool_start_c:
            return
        while temperature > args.cool_resume_c:
            if time.monotonic() - started > 300:
                raise RuntimeError("Cooling timed out")
            time.sleep(1)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
            peak_c = max(peak_c, temperature)
            if temperature >= 84:
                raise RuntimeError("Thermal abort while cooling")

    report = {"status": "running", "seed": args.seed, "requested_steps": args.steps,
              "completed_steps": 0, "config": str(Path(args.config).resolve()),
              "rtol": 0.0, "atol": 0.0, "require_bitwise": True, "maximum_absolute_errors": {},
              "bitwise_equal": {}, "trajectory": [], "promoted": False,
              "graph": args.graph,
              "split_loss": args.split_loss,
              "cache_backward_weights": args.cache_backward_weights,
              "limitations": ["full-model differential test, not a speed benchmark",
                              "single GPU", "uncontrolled clocks"]}

    def compare(name, left, right):
        if left is None or right is None:
            if left is not right:
                raise AssertionError(f"{name}: missing tensor")
            return
        category = name.split(":", 1)[0]
        errors = report["maximum_absolute_errors"]
        equal = report["bitwise_equal"]
        if torch.equal(left, right):
            errors[category] = max(errors.get(category, 0.0), 0.0)
            equal[category] = equal.get(category, True)
            return
        equal[category] = False
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError(f"{name}: nonfinite tensor")
        error = float((left.detach().float() - right.detach().float()).abs().max())
        errors[category] = max(errors.get(category, 0.0), error)
        raise AssertionError(f"{name}: max absolute error {error}")

    def compare_group(category, pairs):
        # One GPU reduction per category, not thousands of host synchronizations.
        present = []
        for name, left, right in pairs:
            if left is None or right is None:
                compare(category + ":" + name, left, right)
            else:
                present.append((name, left, right))
        if not present:
            return
        try:
            compare(category, torch.cat([p[1].detach().reshape(-1) for p in present]),
                    torch.cat([p[2].detach().reshape(-1) for p in present]))
        except AssertionError:
            for name, left, right in present:
                compare(category + ":" + name, left, right)
            raise

    try:
        models = []
        for _ in range(2):
            cool()
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            models.append(_build_qlora_model(spec).train())
        parameters = [{n: p for n, p in model.named_parameters() if p.requires_grad}
                      for model in models]
        if parameters[0].keys() != parameters[1].keys():
            raise AssertionError("Adapter geometry mismatch")
        other_parameters = dict(models[1].named_parameters())
        for name, parameter in models[0].named_parameters():
            if not torch.equal(parameter, other_parameters[name]):
                raise AssertionError(f"Initial model parameter mismatch: {name}")
        for name in parameters[0]:
            if not torch.equal(parameters[0][name], parameters[1][name]):
                raise AssertionError(f"Initial adapter mismatch: {name}")
        if any(hasattr(module, "_molt_original_forward") for module in models[0].modules()):
            raise ValueError("Reference config must not already enable scheduled NF4")
        report["reference_control"] = args.reference_control
        report["analytical_math_override"] = args.analytical_math
        report["loss_backend"] = spec.qlora_loss_backend
        report["sdpa"] = spec.qlora_sdpa_kernel
        report["suffixes"] = args.suffixes
        report["patched_modules"] = 0 if args.reference_control else enable_scheduled_nf4_lora(
            models[1],
            module_suffixes=tuple(args.suffixes.split(",")) if args.suffixes else None,
            cache_backward_weights=args.cache_backward_weights,
        )
        report["scheduled_forward_calls"] = 0
        def count_scheduled(module, inputs, result):
            if type(result.grad_fn).__name__ == "_ScheduledNF4LoRABackward":
                report["scheduled_forward_calls"] += 1
        for module in models[1].modules():
            if hasattr(module, "_molt_original_forward"):
                module.register_forward_hook(count_scheduled)
        report["trainable_parameters"] = sum(p.numel() for p in parameters[0].values())
        optimizers = [_build_qlora_optimizer(model, spec) for model in models]
        losses = [_loss_builder(model, spec, 1) for model in models]
        validation_backends = None
        if args.split_loss:
            import importlib

            kernel_module = importlib.import_module("molt_stream.kernels.frozen_linear_cross_entropy")
            production_dispatch = kernel_module._triton_frozen_head_forward
            separated_control = kernel_module._separated_triton_frozen_head_forward

            def force_backend(loss_function, backend):
                def loss_with_backend(*batch):
                    original = kernel_module._triton_frozen_head_forward
                    try:
                        kernel_module._triton_frozen_head_forward = backend
                        return loss_function(*batch)
                    finally:
                        kernel_module._triton_frozen_head_forward = original
                return loss_with_backend

            # Automatic production dispatch would otherwise send both arms to
            # the split implementation and make this differential gate vacuous.
            losses[0] = force_backend(losses[0], separated_control)
            losses[1] = force_backend(losses[1], production_dispatch)
            validation_backends = (separated_control, production_dispatch)
            report["split_loss_control"] = "separated-buffer"
            report["split_loss_candidate"] = "production-auto-dispatch"
        batcher = MMapTokenBatcher(spec.data, seed=args.seed, device="cuda")
        validation = MMapTokenBatcher(
            replace(spec.data, path=spec.data.validation_path), seed=args.seed + 1, device="cuda")
        val_batch = validation.batch(spec.batch_size)
        validation_batches = [val_batch] + [validation.batch(spec.batch_size)
                                           for _ in range(args.validation_batches - 1)]
        def evaluate():
            totals = [0.0, 0.0]
            for index, validation_batch in enumerate(validation_batches):
                cool()
                if validation_backends is None:
                    values = [
                        _validation_nll(model, spec, validation_batch)
                        for model in models
                    ]
                else:
                    values = []
                    for model, backend in zip(models, validation_backends):
                        original = kernel_module._triton_frozen_head_forward
                        try:
                            kernel_module._triton_frozen_head_forward = backend
                            values.append(_validation_nll(model, spec, validation_batch))
                        finally:
                            kernel_module._triton_frozen_head_forward = original
                compare("validation_batch", torch.tensor(values[0]), torch.tensor(values[1]))
                totals = [total + value for total, value in zip(totals, values)]
                if index % 64 == 0:
                    print(f"Validation pair {index + 1}/{args.validation_batches}", flush=True)
            return [total / args.validation_batches for total in totals]
        initial = evaluate()
        compare("initial_validation", torch.tensor(initial[0]), torch.tensor(initial[1]))
        report["initial_validation"] = initial
        runners = None
        if args.graph:
            from molt_stream.training.static_cuda_graph import StaticCudaMicrobatch
            runners = []
            for arm in range(2):
                cool()
                runners.append(StaticCudaMicrobatch(losses[arm], val_batch,
                       tuple(parameters[arm].values()), joint_forward_backward=True))
                cool()
        for step in range(args.steps):
            cool()
            batch = batcher.batch(spec.batch_size)
            values = []
            for arm in range(2):
                torch.manual_seed(args.seed + step)
                torch.cuda.manual_seed_all(args.seed + step)
                if runners is None:
                    optimizers[arm].zero_grad(set_to_none=True)
                    loss = losses[arm](*batch)
                    loss.backward()
                else:
                    runners[arm].zero_grad()
                    loss = runners[arm].replay(*batch)
                values.append(loss.detach())
            compare_group("gradient", [(name, parameters[0][name].grad, parameters[1][name].grad)
                                       for name in parameters[0]])
            compare("loss", *values)
            for optimizer in optimizers:
                optimizer.step()
            compare_group("adapter", [(name, parameters[0][name], parameters[1][name])
                                      for name in parameters[0]])
            state_groups = {}
            for name in parameters[0]:
                left, right = parameters[0][name], parameters[1][name]
                states = [optimizers[0].state[left], optimizers[1].state[right]]
                if states[0].keys() != states[1].keys():
                    raise AssertionError("Optimizer state keys differ")
                for key in states[0]:
                    state_groups.setdefault(key, []).append((name,
                        torch.as_tensor(states[0][key]), torch.as_tensor(states[1][key])))
            for key, pairs in state_groups.items():
                compare_group("optimizer_" + key, pairs)
            report["completed_steps"] = step + 1
            report["prediction_targets_per_arm"] = (step + 1) * spec.batch_size * spec.data.context_length
            report["trajectory"].append([float(value) for value in values])
            report["peak_sampled_temperature_c"] = peak_c
            if (
                (step + 1) % args.checkpoint_interval == 0
                or step + 1 == args.steps
            ):
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if step % args.progress_interval == 0:
                print(f"PASS update {step + 1}/{args.steps}; loss={float(values[0]):.6f}", flush=True)
        cool()
        final = evaluate()
        compare("final_validation", torch.tensor(final[0]), torch.tensor(final[1]))
        report["final_validation"] = final
        report["validation_targets"] = int(val_batch[1].numel()) * args.validation_batches
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["peak_sampled_temperature_c"] = peak_c
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pynvml.nvmlShutdown()
    print(json.dumps({k: v for k, v in report.items() if k != "trajectory"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
