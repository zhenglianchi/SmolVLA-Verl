#!/bin/bash
# 恢复训练：round 4 起，4 实例 x 4 集，每轮 /train + /clear 后重启 serve 防止 CUDA 上下文劣化
set -uo pipefail
export MUJOCO_GL=egl
export SERVER=http://127.0.0.1:8000
export LOCAL_PYTHON=/home/ubuntu/vla_serve/bin/python
cd /home/ubuntu/SmolVLA-Verl
START_ROUND=${START_ROUND:-4}
END_ROUND=${END_ROUND:-7}
INSTANCES=4
TASKS=(0 1 2 3 4 5 6 7 8 9)
mkdir -p work/logs
for r in $(seq "$START_ROUND" "$END_ROUND"); do
  echo "=== resume round $r/$END_ROUND ==="
  PIDS=()
  for i in $(seq 0 $((INSTANCES-1))); do
    task=${TASKS[$(( (r*INSTANCES + i) % 10 ))]}
    seed=$((20260816 + r*10000 + i*7))
    init_state=$(( (r + i) % 10 ))
    "$LOCAL_PYTHON" scripts/collect_remote.py \
      --server "$SERVER" --suite libero_spatial --task-id "$task" \
      --rollout-n 4 --group-id "r${r}_g${i}" --session-id "r${r}_s${i}" \
      --eta 0.05 --max-steps 280 --action-steps 5 --seed "$seed" \
      --init-state-id "$init_state" \
      > "work/logs/round${r}_i${i}.log" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid"; done
  n_ok=$(grep -hac "success=True" work/logs/round${r}_i*.log | awk '{s+=$1} END{print s+0}')
  echo "  round $r success rate: $n_ok / $((INSTANCES*4)) = $(awk "BEGIN{printf \"%.1f\", $n_ok*100/($INSTANCES*4)}")%"
  TRAIN_RESP=$(curl -s -X POST "$SERVER/train")
  if [[ "$TRAIN_RESP" != *trained* ]]; then
    echo "  ERROR: /train failed: $TRAIN_RESP"
    exit 1
  fi
  echo "  trained: $TRAIN_RESP"
  curl -s -X POST "$SERVER/clear" > /dev/null
  # 重启 serve 加载最新权重，避免 CUDA 上下文劣化
  echo "  restarting serve..."
  bash /home/ubuntu/SmolVLA-Verl/scripts/serve_start.sh
done
echo "RESUME_LOOP_DONE"
