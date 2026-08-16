# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SmolVLA integration for verl-vla.

SmolVLA is a compact flow-matching VLA: a frozen SmolVLM-2 VLM conditions a
trainable flow-matching action expert. The native policy lives in LeRobot and
is kept as the source of truth; this package only adapts it to the verl-vla
training / rollout contracts.

Reference algorithm (FlowGRPO on the flow-matching action expert):
    - FlowVLA-RL: ODE -> SDE marginal-preserving sampler with closed-form
      log-density, then critic-free GRPO (https://github.com/BlackMirean/FlowVLA-RL)
    - Flow-GRPO: "Flow-GRPO: Training Flow Matching Models via Online RL",
      NeurIPS 2025 (arXiv:2505.05470)
"""

from .configuration import SmolVLAConfig
from .modeling import SmolVLAPolicy
from .processor import load_smolvla_processors
from .grpo import compute_group_advantages, grpo_loss
from .sde import marginal_preserving_transition, gaussian_log_prob, sample_transition
from .sde_sampling import SmolVLATrajectory, prepare_policy_prefix, recompute_log_probs, sample_sde_chunk
from .trainable_model import SmolVLATrainableModel

__all__ = [
    "SmolVLAConfig",
    "SmolVLAPolicy",
    "SmolVLATrainableModel",
    "load_smolvla_processors",
    "SmolVLATrajectory",
    "prepare_policy_prefix",
    "recompute_log_probs",
    "sample_sde_chunk",
    "compute_group_advantages",
    "grpo_loss",
]