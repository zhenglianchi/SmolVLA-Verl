#!/bin/bash
# 定时：停训练 + 停服务 -> 官方评估（失败自动重试一次）
LOG=/home/ubuntu/auto_eval.log
echo "[$(date)] auto_eval START" > "$LOG"
pkill -f run_grpo_loop 2>/dev/null
pkill -f collect_remote 2>/dev/null
pkill -9 -f serve_smolvla 2>/dev/null
sleep 5
echo "[$(date)] loop + serve stopped; weights = last completed round (smolvla_grpo)" >> "$LOG"
wait_gpu() {
  for i in $(seq 1 24); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d " MiB")
    [ "${USED:-9999}" -lt 500 ] && return 0
    sleep 15
  done
  return 1
}
run_eval() {
  bash /home/ubuntu/SmolVLA-Verl/scripts/eval_on_server.sh >> "$LOG" 2>&1
}
if wait_gpu; then
  echo "[$(date)] GPU free ($USED MiB); starting official eval (attempt 1)" >> "$LOG"
  if run_eval; then
    echo "[$(date)] EVAL OK" >> "$LOG"
  else
    echo "[$(date)] EVAL attempt 1 FAILED; cleanup + retry" >> "$LOG"
    pkill -9 -f lerobot_eval 2>/dev/null
    sleep 10
    if wait_gpu; then
      echo "[$(date)] GPU free; eval attempt 2" >> "$LOG"
      if run_eval; then
        echo "[$(date)] EVAL OK (attempt 2)" >> "$LOG"
      else
        echo "[$(date)] EVAL FAILED attempt 2" >> "$LOG"
      fi
    else
      echo "[$(date)] GPU never freed after attempt 1" >> "$LOG"
    fi
  fi
else
  echo "[$(date)] GPU never freed; eval skipped" >> "$LOG"
fi
echo "[$(date)] auto_eval DONE" >> "$LOG"