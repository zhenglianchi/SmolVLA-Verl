#!/usr/bin/env bash
# 正式训练：任务轮换全样本 + 16 并行采集实例（360x360）
# 每轮：采集(16组x4=64集) -> /train -> /clear；/train 失败立即停止
set -uo pipefail
export MUJOCO_GL=egl
cd "$(dirname "$0")/.."
SERVER=${SERVER:-http://127.0.0.1:8000}
PY=${LOCAL_PYTHON:-/root/miniconda3/envs/vla_libero312/bin/python}
ROUNDS=${ROUNDS:-5}
INSTANCES=${INSTANCES:-16}
TASKS=(0 1 2 3 4 5 6 7 8 9)
mkdir -p work/logs

for r in $(seq 1 "$ROUNDS"); do
  echo "=== round $r/$ROUNDS (16 instances, task rotation) ==="
  CONCURRENT=${CONCURRENT:-8}   # waves of 8 to bound local RAM at 360x360
  for wave_start in $(seq 0 $CONCURRENT $((INSTANCES-1))); do
    PIDS=()
    for i in $(seq $wave_start $((wave_start+CONCURRENT-1))); do
      [ "$i" -ge "$INSTANCES" ] && break
      task=${TASKS[$(( (r*INSTANCES + i) % 10 ))]}
      seed=$((20260816 + r*10000 + i*7))
      "$PY" scripts/collect_remote.py \
        --server "$SERVER" --suite libero_spatial --task-id "$task" \
        --rollout-n 4 --group-id "r${r}_g${i}" --session-id "r${r}_s${i}" \
        --eta 0.1 --max-steps 280 --action-steps 5 --seed "$seed" \
        > "work/logs/round${r}_i${i}.log" 2>&1 &
      PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
  done
  n_ok=$(grep -hac "success=True" work/logs/round${r}_i*.log | awk '{s+=$1} END{print s+0}')
  echo "  round $r success rate: $n_ok / $((INSTANCES*4)) = $(awk "BEGIN{printf \"%.1f\", $n_ok*100/($INSTANCES*4)}")%"
  TRAIN_RESP=$(curl -s -X POST "$SERVER/train")
  if [[ "$TRAIN_RESP" != *trained* ]]; then
    echo "  ERROR: /train failed: $TRAIN_RESP"
    exit 1
  fi
  echo "  trained: $TRAIN_RESP"
  curl -s -X POST "$SERVER/clear" > /dev/null
done
echo "LOOP_DONE"
