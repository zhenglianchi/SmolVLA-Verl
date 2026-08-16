# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Load and initialize native LeRobot SmolVLA processors."""

from __future__ import annotations

from pathlib import Path

from lerobot.datasets.utils import load_stats
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

__all__ = ["load_smolvla_processors"]


def load_smolvla_processors(config, model_path: str | Path, *, dataset_root: str | Path | None = None):
    """Load the serialized SmolVLA pre/post processors from a trained checkpoint.

    LeRobot stores normalization and language-tokenization steps outside the
    policy config. A trained SmolVLA checkpoint always carries
    ``policy_preprocessor.json`` + ``policy_postprocessor.json`` sidecars (plus
    their safetensors), so the default path is a direct reload. Config-only
    initialization requires the training dataset statistics once.

    Args:
        config: The native ``SmolVLAConfig``.
        model_path: Directory containing the checkpoint (and processor sidecars).
        dataset_root: LeRobot dataset root, required only for config-only init.
    """

    model_path = Path(model_path)
    preprocessor_path = model_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    postprocessor_path = model_path / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    processor_files = (preprocessor_path.is_file(), postprocessor_path.is_file())
    if any(processor_files) and not all(processor_files):
        raise FileNotFoundError(
            f"Incomplete native SmolVLA processor artifacts in {model_path}: both "
            f"{preprocessor_path.name} and {postprocessor_path.name} are required"
        )
    if all(processor_files):
        return make_pre_post_processors(config, pretrained_path=str(model_path))
    if dataset_root is None:
        raise FileNotFoundError(
            f"Native SmolVLA processor artifacts are missing from {model_path}. "
            "Set model.adapter.processor_dataset_root when initializing from config."
        )
    stats = load_stats(Path(dataset_root))
    if stats is None:
        raise FileNotFoundError(f"LeRobot dataset statistics are missing from {Path(dataset_root) / 'meta/stats.json'}")
    return make_pre_post_processors(config, dataset_stats=stats)