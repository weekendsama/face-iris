from __future__ import annotations

import os
from dataclasses import replace

from .config import DEFAULT_CONFIG, ModelConfig


PRESET_DEFAULT = "baseline"


def _build_presets() -> dict[str, ModelConfig]:
    baseline = DEFAULT_CONFIG
    fast_debug = replace(
        baseline,
        train_steps=2,
        eval_warmup_steps=2,
        eval_batches=4,
        batch_size=4,
        sampler_identities_per_batch=2,
        dsh_num_clusters=2,
        dsh_kmeans_iters=2,
    )
    wider_hash = replace(
        baseline,
        shared_hash_bits=512,
        shared_embedding_dim=384,
        block_size=64,
    )
    stable_dsh = replace(
        baseline,
        dsh_bank_per_class=5,
        dsh_min_samples_per_class=3,
        dsh_kmeans_iters=5,
    )
    return {
        "baseline": baseline,
        "fast_debug": fast_debug,
        "wider_hash": wider_hash,
        "stable_dsh": stable_dsh,
    }


PRESETS = _build_presets()


def get_config(preset_name: str | None = None) -> tuple[ModelConfig, str]:
    name = preset_name or os.getenv("EXPERIMENT_PRESET") or PRESET_DEFAULT
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name], name
