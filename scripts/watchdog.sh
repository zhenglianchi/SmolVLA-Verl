#!/bin/bash
# 看门狗：serve/训练循环意外退出时自动恢复（断点续训），直到 STOP_AT 停训；评估期间不干预
LOG=/home/ubuntu/runs/watchdog.log
STOP_AT="2026-08-19 17:00"
STOP_EPOCH=$(date -d "$STOP_AT" +%s 2>/dev/null || echo 0)
mkdir -p /home/ubuntu/runs
if [ "$STOP_EPOCH" -gt 0 ] && [ "$(date +%s)" -ge "$STOP_EPOCH" ]; then
  exit 0
fi
if pgrep -f eval_task_loop > /dev/null || pgrep -f lerobot_eval > /dev/null; then
  exit 0
fi
if [ -f /home/ubuntu/grpo_opt.log ] && grep -q "OPT_ALL_DONE" /home/ubuntu/grpo_opt.log; then
  exit 0
fi
if ! curl -s -m 5 http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"'; then
  echo "[$(date)] serve dead, restarting" >> "$LOG"
  if [ -f /home/ubuntu/runs/smolvla_grpo/model.safetensors ]; then
    CHECKPOINT=/home/ubuntu/runs/smolvla_grpo bash /home/ubuntu/SmolVLA-Verl/scripts/serve_start.sh >> "$LOG" 2>&1 || true
  else
    CHECKPOINT=/home/ubuntu/models/smolvla_libero bash /home/ubuntu/SmolVLA-Verl/scripts/serve_start.sh >> "$LOG" 2>&1 || true
  fi
fi
if ! pgrep -f run_loop_opt > /dev/null; then
  LAST=1
  [ -f /home/ubuntu/SmolVLA-Verl/work/logs/last_round ] && LAST=$(cat /home/ubuntu/SmolVLA-Verl/work/logs/last_round)
  START=$((LAST + 1))
  echo "[$(date)] loop dead, restarting from round $START" >> "$LOG"
  cd /home/ubuntu/SmolVLA-Verl
  setsid nohup env ROUNDS=15 INSTANCES=12 GRPO_LR=5e-6 GRPO_STEPS=1 BATCH_SIZE=32 START_ROUND=$START STOP_AT="$STOP_AT" \
    bash scripts/run_loop_opt.sh >> /home/ubuntu/grpo_opt.log 2>&1 < /dev/null &
  echo "[$(date)] restart issued, pid $!" >> "$LOG"
fi
exit 0
