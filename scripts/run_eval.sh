#!/usr/bin/env bash
# 评估：官方协议 10 trials/任务（eval seed 1000），默认 libero_spatial
set -xeuo pipefail
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
ENV=${VLA_ENV:-/home/ubuntu/vla_verl}
POLICY=${POLICY:-/home/ubuntu/runs/smolvla_grpo}
TASK=${TASK:-libero_spatial}
OUT=${OUT:-/home/ubuntu/results/grpo_${TASK}}
"$ENV/bin/python" -m lerobot.scripts.lerobot_eval \
  --policy.path="$POLICY" \
  --env.type=libero --env.task="$TASK" \
  --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false \
  --policy.device=cuda \
  --env.camera_name_mapping='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}' \
  --rename_map='{"observation.images.camera1":"observation.images.image","observation.images.camera2":"observation.images.image2"}' \
  --policy.empty_cameras=1 --seed=1000 \
  --output_dir="$OUT"
