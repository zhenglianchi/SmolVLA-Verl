#!/usr/bin/env bash
# 小样本冒烟：1 round，验证能训练（可续训、能存权重）
set -xeuo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
ENV=${VLA_ENV:-/home/ubuntu/vla_verl}
cd "$(dirname "$0")/.."
"$ENV/bin/python" -m smolvla_verl.trainer.grpo_libero \
  --checkpoint "${CHECKPOINT:-/home/ubuntu/models/smolvla_libero}" \
  --suite libero_spatial --task-ids 0 \
  --rollout-n 2 --groups-per-round 1 --rounds 1 \
  --num-runners 1 --max-steps 30 --chunk-size 10 \
  --eta 0.4 --lr 1e-6 --clip-epsilon 0.2 --kl-beta 0.01 \
  --save-dir "${SAVE_DIR:-/home/ubuntu/runs/smolvla_grpo_smoke}"
