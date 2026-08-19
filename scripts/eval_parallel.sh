#!/bin/bash
# 官方评估：10 任务分两波并行（每波 5 个独立进程），逐任务独立目录，seed=1000 确定性
# 用法：OUT=/home/ubuntu/results/grpo_final_pertask POLICY=/home/ubuntu/runs/smolvla_grpo bash scripts/eval_parallel.sh
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
OUT=${OUT:-/home/ubuntu/results/grpo_final_pertask}
mkdir -p "$OUT"
: > "$OUT/parallel_status.log"

run_task() {
  tid=$1
  TDIR="$OUT/task_$tid"
  rm -rf "$TDIR"
  "$ENV/bin/python" -m lerobot.scripts.lerobot_eval \
    --policy.path="$POLICY" --env.type=libero --env.task="$TASK" --env.task_ids="[$tid]" \
    --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false \
    --policy.device=cuda --policy.n_action_steps=5 \
    --env.camera_name_mapping='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}' \
    --rename_map='{"observation.images.camera1":"observation.images.image","observation.images.camera2":"observation.images.image2"}' \
    --seed=1000 --output_dir="$TDIR" > "$OUT/task_$tid.log" 2>&1
  echo "task $tid done rc=$?" >> "$OUT/parallel_status.log"
}

for wave in 0 1; do
  PIDS=()
  for tid in $(seq $((wave*5)) $((wave*5+4))); do
    run_task $tid &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait $p; done
done
echo "PARALLEL_EVAL_DONE" >> "$OUT/parallel_status.log"
