#!/usr/bin/env python
"""Stage-A smoke: build SmolVLA trainable model through verl-vla's builder.

Validates: VLAModelConfig architecture detection -> build_vla_model ->
processor loading -> SmolVLAPolicy.from_pretrained -> SmolVLATrainableModel.
"""
from __future__ import annotations

import torch

from verl_vla.models import build_vla_model
from verl_vla.workers.config.model import VLAModelConfig

CHECKPOINT = "/root/vla_libero/models/smolvla_libero"


def main() -> None:
    cfg = VLAModelConfig(path=CHECKPOINT, use_shm=False)
    print("native_architecture:", cfg.native_architecture)
    assert cfg.native_architecture == "smolvla", cfg.native_architecture

    model = build_vla_model(cfg, torch_dtype=torch.bfloat16)
    print("model type:", type(model).__name__)
    print("has sample_sde_chunk:", hasattr(model, "sample_sde_chunk"))
    print("has flow_log_prob:", hasattr(model, "flow_log_prob"))
    print("action_dim:", model.action_dim)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"params trainable/total: {trainable/1e6:.3f}M / {total/1e6:.3f}M")

    # processors loaded?
    print("preprocessor steps:", [type(s).__name__ for s in model.preprocessor.steps])
    print("postprocessor steps:", [type(s).__name__ for s in model.postprocessor.steps])
    print("SMOKE_BUILD_OK")


if __name__ == "__main__":
    main()