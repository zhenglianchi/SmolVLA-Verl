#!/bin/bash
# 逐任务评估（10 任务 x 10 集），每任务独立进程/结果，每任务最多重试 2 次
# 抗间歇性 CUDA 错误：崩了只重跑该任务，已完成任务的结果保留
set -uo pipefail
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV=/home/ubuntu/vla_serve
POLICY=${POLICY:-/home/ubuntu/runs/smolvla_grpo}
TASK=${TASK:-libero_spatial}
OUT=${OUT:-/home/ubuntu/results/grpo_${TASK}_pertask}
mkdir -p "$OUT"
wait_gpu() {
  for i in $(seq 1 20); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d " MiB")
    [ "${USED:-9999}" -lt 300 ] && return 0
    sleep 10
  done
  return 1
}
for tid in $(seq 0 9); do
  TDIR="$OUT/task_$tid"
  if [ -f "$TDIR/eval_info.json" ]; then
    echo "[eval-task] task $tid already done, skip"
    continue
  fi
  ok=0
  for attempt in 1 2 3; do
    echo "[eval-task] task $tid attempt $attempt start $(date)"
    rm -rf "$TDIR"
    "$ENV/bin/python" -m lerobot.scripts.lerobot_eval \
      --policy.path="$POLICY" --env.type=libero --env.task="$TASK" --env.task_ids="[$tid]" \
      --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false \
      --policy.device=cuda --policy.n_action_steps=5 \
      --env.camera_name_mapping='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}' \
      --rename_map='{"observation.images.camera1":"observation.images.image","observation.images.camera2":"observation.images.image2"}' \
      --seed=1000 --output_dir="$TDIR" >> "$OUT/task_$tid.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$TDIR/eval_info.json" ]; then
      echo "[eval-task] task $tid attempt $attempt OK $(date)"
      ok=1
      break
    fi
    echo "[eval-task] task $tid attempt $attempt FAILED rc=$rc $(date)"
    pkill -9 -f lerobot_eval 2>/dev/null
    sleep 5
    wait_gpu || echo "[eval-task] warning: GPU not free"
  done
  if [ $ok -eq 0 ]; then
    echo "[eval-task] task $tid GAVE UP after 3 attempts"
  fi
done
echo "[eval-task] aggregating"
"$ENV/bin/python" /home/ubuntu/SmolVLA-Verl/scripts/parse_eval_results.py "$OUT" | tee /home/ubuntu/results/grpo_${TASK}_pertask_summary.txt
echo "[eval-task] EVAL_OK"