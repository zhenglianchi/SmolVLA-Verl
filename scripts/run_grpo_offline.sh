#!/usr/bin/env bash
# 服务器离线 GRPO 训练（无 env，只重打分+训练）
set -xeuo pipefail
ulimit -n 65535
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
ENV=${VLA_ENV:-/home/ubuntu/vla_verl}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
"$ENV/bin/python" -m smolvla_verl.trainer.grpo_offline \
  --trajectories "${TRAJ:-/home/ubuntu/trajectories/traj.pkl}" \
  --checkpoint "${CHECKPOINT:-/home/ubuntu/models/smolvla_libero}" \
  --lr "${LR:-1e-6}" --clip-epsilon 0.2 --kl-beta "${KL_BETA:-0.01}" \
  --rounds "${ROUNDS:-10}" --epochs-per-round "${EPOCHS:-1}" \
  --save-dir "${SAVE_DIR:-/home/ubuntu/runs/smolvla_grpo}" \
  --resume
