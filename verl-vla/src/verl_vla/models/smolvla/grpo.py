# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Critic-free GRPO objectives for flow-matching denoising trajectories.

Ported from FlowVLA-RL (MIT, https://github.com/BlackMirean/FlowVLA-RL)
``src/flowvla_rl/grpo.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["compute_group_advantages", "k3_kl_estimate", "grpo_loss"]


def compute_group_advantages(
    rewards: Tensor,
    *,
    use_std: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Center complete-episode rewards within each reset-matched group.

    Input shape is ``(number_of_groups, group_size)``. The default deliberately
    does not divide by standard deviation (the Dr.GRPO convention).
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (groups, group_size>=2)")
    advantages = rewards.float() - rewards.float().mean(dim=1, keepdim=True)
    if use_std:
        advantages = advantages / (rewards.float().std(dim=1, keepdim=True, unbiased=False) + eps)
    return advantages.detach()


def k3_kl_estimate(logp: Tensor, ref_logp: Tensor) -> Tensor:
    """Non-negative k3 estimate of KL(policy || reference).

    With ``log_ratio = log p_ref - log p_policy``, k3 is
    ``exp(log_ratio) - log_ratio - 1``.
    """
    log_ratio = (ref_logp.detach().float() - logp.float()).clamp(-10.0, 10.0)
    return torch.exp(log_ratio) - log_ratio - 1.0


def grpo_loss(
    logp: Tensor,
    old_logp: Tensor,
    ref_logp: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float,
    kl_beta: float,
    valid_steps: Tensor | None = None,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute per-denoise-step clipped GRPO plus in-loss reference KL.

    ``logp`` has shape ``(episodes, control_chunks, denoise_steps)``. One scalar
    episode advantage is broadcast across both inner axes. There is no
    transition probability between adjacent control chunks.

    ``sample_weights`` (optional, shape ``(episodes,)``) reweights each episode
    inside the mean. It is used for episode-balanced chunk weighting (failed
    episodes are longer and otherwise dominate the objective) combined with an
    optional reward-to-go chunk discount.
    """
    if logp.ndim != 3:
        raise ValueError("logp must have shape (episodes, chunks, denoise_steps)")
    if old_logp.shape != logp.shape or ref_logp.shape != logp.shape:
        raise ValueError("old_logp and ref_logp must match logp")
    if advantages.shape != (logp.shape[0],):
        raise ValueError("advantages must contain one scalar per episode")
    if clip_epsilon < 0 or kl_beta < 0:
        raise ValueError("clip_epsilon and kl_beta must be non-negative")
    if sample_weights is not None and sample_weights.shape != (logp.shape[0],):
        raise ValueError(
            f"sample_weights must have shape {(logp.shape[0],)}, got {tuple(sample_weights.shape)}"
        )

    if valid_steps is None:
        valid = torch.ones_like(logp, dtype=torch.bool)
    else:
        try:
            valid = torch.broadcast_to(valid_steps.to(device=logp.device, dtype=torch.bool), logp.shape)
        except RuntimeError as exc:
            raise ValueError("valid_steps is not broadcastable to logp") from exc
    valid_f = valid.float()
    if sample_weights is None:
        sample_w = torch.ones_like(valid_f[:, :1, :1])
    else:
        sample_w = sample_weights.detach().to(
            device=logp.device, dtype=torch.float32
        )[:, None, None]
    weighted_valid = valid_f * sample_w
    # NOTE: with per-sample (episode-balanced) weights the weighted sum is far
    # below 1 (e.g. 0.01-0.3 per chunk), so a floor of 1.0 would silently
    # inflate the loss/gradient by ~1/sum(w) AND corrupt the ratio_mean metric
    # (this was the "first-forward ratio drifts" false alarm). Only guard
    # against an exactly-zero denominator.
    denominator = weighted_valid.sum().clamp_min(1e-12)

    log_ratio = (logp.float() - old_logp.detach().float()).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    advantage = advantages.detach().float()[:, None, None]
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    pg_loss = -(torch.minimum(unclipped, clipped) * weighted_valid).sum() / denominator

    kl = k3_kl_estimate(logp, ref_logp)
    kl_loss = (kl * weighted_valid).sum() / denominator
    loss = pg_loss + float(kl_beta) * kl_loss
    if not torch.isfinite(loss):
        zero = torch.zeros_like(loss)
        metrics = {k: zero.detach() for k in (
            "loss", "pg_loss", "kl", "ratio_mean", "clip_fraction", "advantage_mean", "advantage_std")}
        return zero, metrics
    metrics = {
        "loss": loss.detach(),
        "pg_loss": pg_loss.detach(),
        "kl": kl_loss.detach(),
        "ratio_mean": (ratio.detach() * weighted_valid).sum() / denominator,
        "clip_fraction": (((ratio - 1.0).abs() > clip_epsilon).float() * weighted_valid).sum()
        / denominator,
        "advantage_mean": advantages.detach().float().mean(),
        "advantage_std": advantages.detach().float().std(unbiased=False),
    }
    return loss, metrics
