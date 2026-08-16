# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""verl-vla adapter around LeRobot's native SmolVLAPolicy (flow-matching VLA)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from torch import Tensor, nn
from verl import DataProto

from ..base import SupportSFTTraining, TrainableVLAModelBase
from .grpo import grpo_loss
from .modeling import SmolVLAPolicy
from .sde_sampling import (
    SmolVLATrajectory,
    prepare_policy_prefix,
    recompute_log_probs,
    rollout_autocast,
    sample_sde_chunk as sample_sde_chunk_fn,
)

__all__ = ["SmolVLATrainableModel"]


class SmolVLATrainableModel(TrainableVLAModelBase, SupportSFTTraining):
    """Trainable wrapper for the LeRobot SmolVLA policy.

    SmolVLA = frozen SmolVLM-2 VLM + trainable flow-matching action expert.
    The native policy stays the source of truth for preprocessing, sampling and
    checkpoints; this wrapper only adapts the verl-vla ``DataProto`` observation
    contract to the native batch contract (observation keys + ``task``
    instruction) and forwards to the native APIs.

    Exposed capabilities:
      * ``select_action`` / ``predict_action_chunk``: rollout & inference.
      * ``sft_loss``: native flow-matching supervised loss via ``policy.forward``.
      * ``flow_log_prob`` / ``flow_policy_loss``: FlowGRPO hooks. These are the
        RL-milestone extension points (ODE->SDE marginal-preserving sampler with
        closed-form log-density, ported from FlowVLA-RL / Flow-GRPO).
    """

    def __init__(
        self,
        policy: SmolVLAPolicy,
        *,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        adapter_config: dict | None = None,
    ) -> None:
        super().__init__(policy=policy)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        config = dict(adapter_config or {})
        # FlowGRPO sampling knobs (consumed by the RL rollout; safe defaults
        # matching the native SFT checkpoint: 10 flow steps, ODE-equivalent).
        self.flow_steps = int(config.pop("flow_steps", 10))
        self.sde_sigma = float(config.pop("sde_sigma", 1.0))
        self.sde_eta = float(config.pop("sde_eta", 1.0))
        if config:
            raise ValueError(f"Unsupported SmolVLA adapter fields: {sorted(config)}")

        SupportSFTTraining.__init__(self, adapter_config or {})
        self.config = policy.config

    # ------------------------------------------------------------------ #
    # Native batch construction
    # ------------------------------------------------------------------ #
    def _policy_batch(self, obs: DataProto) -> dict:
        """Translate a verl-vla observation batch into a native SmolVLA batch.

        Native SmolVLA expects observation keys (images / state) plus a ``task``
        instruction string; the native preprocessor then normalizes state/action
        and tokenizes the language instruction. Observation tensors are passed
        through untouched (the environment already emits them in the format the
        native pipeline expects, e.g. uint8 images).
        """
        batch: dict = {}
        for key in self.policy.config.input_features:
            if key in obs.batch:
                batch[key] = obs.batch[key]
        task = None
        if obs.non_tensor_batch is not None:
            task = obs.non_tensor_batch.get("task")
        if task is None and "task" in obs.batch:
            task = obs.batch["task"]
        if task is None:
            raise ValueError("SmolVLA requires an instruction ('task') in the observation batch")
        if isinstance(task, np.ndarray):
            # LeRobot's tokenizer processor expects a str or list[str] in
            # complementary_data["task"], not an ndarray.
            task = task.tolist()
        batch["task"] = task
        return batch

    # ------------------------------------------------------------------ #
    # Rollout / inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, obs: DataProto, **kwargs) -> Tensor:
        """Preprocess -> native ``select_action`` -> postprocess (env action)."""
        batch = self.preprocessor(self._policy_batch(obs))
        action = self.policy.select_action(batch, **kwargs)
        return self.postprocessor(action)

    @torch.no_grad()
    def predict_action_chunk(self, obs: DataProto, **kwargs) -> Tensor:
        """Preprocess -> native ``predict_action_chunk`` (full action chunk)."""
        batch = self.preprocessor(self._policy_batch(obs))
        return self.policy.predict_action_chunk(batch, **kwargs)

    def reset(self) -> None:
        self.policy.reset()

    # ------------------------------------------------------------------ #
    # SFT contract (flow-matching supervised loss)
    # ------------------------------------------------------------------ #
    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: nn.Module,
        actions: dict[str, Tensor],
        valids: Tensor,
        action_mask: Tensor | None = None,
        target_values: Tensor | None = None,
    ) -> Tensor:
        """Native flow-matching loss.

        The raw (unnormalized) action is added *before* the preprocessor so the
        normalizer step maps it to normalized space, matching the native
        LeRobot training batch contract (obs + ``action`` + ``action_is_pad``).
        """
        del tokenizer, target_values
        action = actions[ACTION]
        if action.ndim != 3:
            raise ValueError(f"SmolVLA SFT actions must have shape (batch, chunk, action_dim), got {action.shape}")
        batch = self._policy_batch(obs)
        batch[ACTION] = action
        if action_mask is not None:
            batch["action_is_pad"] = action_mask.to(torch.bool)
        batch = self.preprocessor(batch)
        loss, _ = self.policy.forward(batch)
        self.sft_metrics["flow_matching_loss"] = loss.detach()
        return loss

    # ------------------------------------------------------------------ #
    # FlowGRPO hooks
    # ------------------------------------------------------------------ #
    def rollout_context(self):
        """Autocast context shared by collection and rescoring.

        Collection and rescoring MUST use the identical numeric path (bf16 on
        CUDA, fp32 on CPU): any drift between the two log-prob estimates pushes
        the first-forward ratio away from 1 and silently kills PPO-style
        clipping (see ``sde_sampling`` docstring).
        """
        return rollout_autocast(self.policy.device if hasattr(self.policy, "device") else next(self.policy.parameters()).device)

    @property
    def action_dim(self) -> int:
        return int(self.policy.config.output_features[ACTION].shape[0])

    @torch.no_grad()
    def sample_sde_chunk(
        self,
        obs: DataProto,
        *,
        eta: float | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> SmolVLATrajectory:
        """Sample one flow-matching chunk with density under the SDE sampler."""
        batch = self.preprocessor(self._policy_batch(obs))
        prefix_pad_masks, prefix_cache = prepare_policy_prefix(self.policy, batch)
        return sample_sde_chunk_fn(
            self.policy.model,
            prefix_pad_masks,
            prefix_cache,
            action_dim=self.action_dim,
            eta=float(self.sde_eta if eta is None else eta),
            noise=noise,
            generator=generator,
        )

    # NOTE: rescoring must be differentiable (velocity recomputation flows
    # through the policy weights for the GRPO policy-gradient step). Only
    # collection-time sampling (sample_sde_chunk) is no_grad.
    def flow_log_prob(
        self,
        obs: DataProto,
        trajectory: SmolVLATrajectory,
        *,
        valid_positions: Tensor | None = None,
    ) -> Tensor:
        """Rescore a fixed trajectory under current weights (old-policy logp)."""
        batch = self.preprocessor(self._policy_batch(obs))
        prefix_pad_masks, prefix_cache = prepare_policy_prefix(self.policy, batch)
        return recompute_log_probs(
            self.policy.model,
            prefix_pad_masks,
            prefix_cache,
            trajectory,
            valid_positions=valid_positions,
        )

    def flow_policy_loss(
        self,
        logp: Tensor,
        old_logp: Tensor,
        ref_logp: Tensor,
        advantages: Tensor,
        *,
        clip_epsilon: float,
        kl_beta: float,
        valid_steps: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """FlowGRPO policy loss: clipped ratio + in-loss reference KL."""
        return grpo_loss(
            logp,
            old_logp,
            ref_logp,
            advantages,
            clip_epsilon=clip_epsilon,
            kl_beta=kl_beta,
            valid_steps=valid_steps,
        )

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #
    def save_pretrained(self, save_directory, *args, state_dict=None, **kwargs) -> None:
        del args
        self.export_policy(save_directory, state_dict=state_dict, **kwargs)

    def export_policy(self, output_dir: str | Path, *, state_dict=None, **kwargs) -> None:
        """Export the native policy and its processors (LeRobot-native format)."""
        if kwargs:
            raise TypeError(f"Unsupported SmolVLA export options: {sorted(kwargs)}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if state_dict is None:
            self.policy.save_pretrained(output_dir)
        else:
            self.policy.save_pretrained(output_dir, state_dict=self.extract_policy_state_dict(state_dict))
        self.preprocessor.save_pretrained(output_dir)
        self.postprocessor.save_pretrained(output_dir)