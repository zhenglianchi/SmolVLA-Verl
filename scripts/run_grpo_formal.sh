#!/usr/bin/env bash
# 全样本训练：rollout.n=4、多 runner 并行、10 轮、续训、单权重夹覆盖
set -xeuo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
ENV=${VLA_ENV:-/home/ubuntu/vla_verl}
cd "$(dirname "$0")/.."
"$ENV/bin/python" -m smolvla_verl.trainer.grpo_libero \
  --checkpoint "${CHECKPOINT:-/home/ubuntu/models/smolvla_libero}" \
  --suite libero_spatial --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --rollout-n 4 --groups-per-round "${GROUPS_PER_ROUND:-8}" --rounds "${ROUNDS:-10}" \
  --num-runners "${NUM_RUNNERS:-4}" --max-steps 280 --chunk-size 50 \
  --eta 0.4 --lr 1e-6 --clip-epsilon 0.2 --kl-beta 0.01 \
  --save-dir "${SAVE_DIR:-/home/ubuntu/runs/smolvla_grpo}" \
  --resume
