#!/bin/bash
# 优化版 GRPO（verl 式批量 + 固定 base 参考锚定）：12实例x4=48集/轮，steps=3，lr=5e-6，batch=32
# STOP_AT 时间截止 + START_ROUND 断点续训 + last_round 进度 + 最佳权重备份
set -uo pipefail
export MUJOCO_GL=egl
export SERVER=http://127.0.0.1:8000
export LOCAL_PYTHON=/home/ubuntu/vla_serve/bin/python
cd "$(dirname "$0")/.."
ROUNDS=${ROUNDS:-60}
INSTANCES=${INSTANCES:-12}
GRPO_LR=${GRPO_LR:-5e-6}
GRPO_STEPS=${GRPO_STEPS:-3}
BATCH_SIZE=${BATCH_SIZE:-32}
START_ROUND=${START_ROUND:-1}
STOP_AT=${STOP_AT:-"2026-08-18 16:30"}
STOP_EPOCH=$(date -d "$STOP_AT" +%s 2>/dev/null || echo 0)
TASKS=(0 1 2 3 4 5 6 7 8 9)
BEST_RATE=0
BEST_DIR=/home/ubuntu/runs/smolvla_best
mkdir -p work/logs
for r in $(seq "$START_ROUND" "$ROUNDS"); do
  if [ "$STOP_EPOCH" -gt 0 ] && [ "$(date +%s)" -ge "$STOP_EPOCH" ]; then
    echo "=== reached stop time $STOP_AT, stopping training at round $r ==="
    break
  fi
  echo "=== opt round $r/$ROUNDS ($INSTANCES instances x 4, lr=$GRPO_LR steps=$GRPO_STEPS batch=$BATCH_SIZE, stop=$STOP_AT) ==="
  PIDS=()
  for i in $(seq 0 $((INSTANCES-1))); do
    task=${TASKS[$(( (r*INSTANCES + i) % 10 ))]}
    seed=$((20260901 + r*10000 + i*7))
    "$LOCAL_PYTHON" scripts/collect_remote.py \
      --server "$SERVER" --suite libero_spatial --task-id "$task" \
      --rollout-n 4 --group-id "r${r}_g${i}" --session-id "r${r}_s${i}" \
      --eta 0.1 --max-steps 280 --action-steps 5 --seed "$seed" \
      > "work/logs/opt_r${r}_i${i}.log" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid"; done
  n_ok=$(grep -hac "success=True" work/logs/opt_r${r}_i*.log | awk '{s+=$1} END{print s+0}')
  echo "  round $r success rate: $n_ok / $((INSTANCES*4)) = $(awk "BEGIN{printf \"%.1f\", $n_ok*100/($INSTANCES*4)}")%"
  TRAIN_RESP=$(curl -s -X POST "$SERVER/train?lr=$GRPO_LR&steps=$GRPO_STEPS&batch_size=$BATCH_SIZE")
  if [[ "$TRAIN_RESP" != *trained* ]]; then
    echo "  ERROR: /train failed: $TRAIN_RESP"
    exit 1
  fi
  echo "  trained: $TRAIN_RESP"
  curl -s -X POST "$SERVER/clear" > /dev/null
  echo "$r" > work/logs/last_round
  if [ "$n_ok" -gt "$BEST_RATE" ]; then
    BEST_RATE=$n_ok
    rm -rf "$BEST_DIR"
    cp -r /home/ubuntu/runs/smolvla_grpo "$BEST_DIR"
    echo "  NEW BEST round $r ($n_ok/$((INSTANCES*4))) -> $BEST_DIR"
  fi
  echo "  restarting serve with latest weights..."
  CHECKPOINT=/home/ubuntu/runs/smolvla_grpo bash /home/ubuntu/SmolVLA-Verl/scripts/serve_start.sh
done
echo "OPT_LOOP_DONE"
echo "=== starting official eval (10 tasks x 10 trials) on final weights ==="
OUT=/home/ubuntu/results/grpo_opt_libero_spatial_pertask POLICY=/home/ubuntu/runs/smolvla_grpo \
  bash /home/ubuntu/SmolVLA-Verl/scripts/eval_task_loop.sh
echo "OPT_ALL_DONE"