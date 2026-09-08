"""Make the generated beginner config explicit and reproducible."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update({
        "batch_size": 1,
        "gradient_accumulation": 1,
        "max_steps": 20,
        "evaluation_interval": 20,
        "learning_rate": 0.0002,
        "seed": 1337,
        "activation_checkpointing": False,
        "qlora_fused_optimizer": True,
        "qlora_autocast": True,
        "qlora_loss_chunk_size": 96,
        "qlora_precompute_head_gradient": True,
        "qlora_cache_frozen_head": True,
        "qlora_restore_tied_embedding_bf16": True,
        "qlora_frozen_rmsnorm_bf16": True,
        "qlora_joint_cuda_graph": True,
        "qlora_validation_batches": 2,
        "thermal_target_c": 80.0,
        "thermal_microbatch_guard_c": 80.0,
        "thermal_abort_c": 84.0,
        "thermal_startup_max_c": 55.0,
        "thermal_startup_dwell_seconds": 5.0,
        "thermal_startup_timeout_seconds": 300.0,
    })
    value.setdefault("stream", {})["cuda_graphs"] = True
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
