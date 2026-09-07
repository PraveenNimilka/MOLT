"""Text-facing adapter for checkpoint generation."""
from pathlib import Path

def generate_text(run: str, prompt: str, tokenizer_path: str | None,
                  max_new_tokens: int) -> dict[str, object]:
    from transformers import AutoTokenizer
    from molt_stream.core.specs import load_spec
    from molt_stream.training.engine import generate_run
    spec = load_spec(Path(run) / "spec.resolved.json")
    location = tokenizer_path or spec.base_model
    if not location:
        raise ValueError("Text generation requires --tokenizer for a pretrained-from-scratch run")
    if not prompt or max_new_tokens < 1:
        raise ValueError("Provide a nonempty prompt and positive max_new_tokens")
    # Security boundary: location is local and network access is disabled.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        location, local_files_only=True
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    value = generate_run(run, tokens, max_new_tokens)
    return {"text": tokenizer.decode(value, skip_special_tokens=True), "token_ids": value}
