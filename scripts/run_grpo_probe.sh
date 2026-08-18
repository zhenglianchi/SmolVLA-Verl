#!/usr/bin/env bash
ulimit -n 65535
# 预正式探针：1 轮、1 组、rollout_n=4、280 步、action_steps=5 —— 验证有成功轨迹 + 计时
set -xeuo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
ENV=${VLA_ENV:-/home/ubuntu/vla_verl}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
"$ENV/bin/python" -m smolvla_verl.trainer.grpo_libero \
  --checkpoint "${CHECKPOINT:-/home/ubuntu/models/smolvla_libero}" \
  --suite libero_spatial --task-ids 0 \
  --rollout-n 4 --groups-per-round 1 --rounds 1 \
  --num-runners "${NUM_RUNNERS:-1}" --max-steps 280 --chunk-size 10 --action-steps 5 \
  --eta "${ETA:-0.1}" --lr 1e-6 --clip-epsilon 0.2 --kl-beta 0.01 \
  --save-dir "${SAVE_DIR:-/home/ubuntu/runs/smolvla_grpo_probe}"
