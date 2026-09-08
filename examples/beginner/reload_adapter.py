"""Load a MOLT PEFT export and run one deterministic adapter inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--prompt", default="Return only the MOLT demo status code."
    )
    args = parser.parse_args()
    model_path = str(Path(args.model).resolve())
    adapter_path = str(Path(args.adapter).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda")
    model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    print(f"Adapter loaded: {adapter_path}")
    print(f"Response: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
