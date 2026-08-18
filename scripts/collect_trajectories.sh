#!/usr/bin/env bash
# 本地采集轨迹（lerobot 0.6.0，官方协议环境）→ work/trajectories/
set -euo pipefail
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
ENV=${LOCAL_ENV:-/root/miniconda3/envs/vla_libero312}
cd "$(dirname "$0")/.."
"$ENV/bin/python" scripts/collect_trajectories.py \
  --checkpoint "${CHECKPOINT:-/root/vla_libero/models/smolvla_libero}" \
  --suite libero_spatial --task-id "${TASK_ID:-0}" \
  --rollout-n "${ROLLOUT_N:-4}" --groups "${GROUPS:-1}" \
  --eta "${ETA:-0.1}" --max-steps "${MAX_STEPS:-280}" \
  --action-steps "${ACTION_STEPS:-5}" --chunk-size "${CHUNK_SIZE:-10}" \
  --seed "${SEED:-20260816}" \
  --out "${OUT:-work/trajectories/traj.pkl}"
