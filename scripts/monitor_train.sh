#!/usr/bin/env bash
# 持续监控训练日志，短间隔，实时打印新进度
LOG=${1:-/home/ubuntu/runs/smolvla_grpo/train.log}
LAST=0
while true; do
  if [ -f "$LOG" ]; then
    LINES=$(wc -l < "$LOG")
    if [ "$LINES" -gt "$LAST" ]; then
      tail -n +$((LAST+1)) "$LOG" | grep -aE "round|rollout|train|save|TRAIN|ERROR|Traceback|success" | tail -6
      LAST=$LINES
    fi
  fi
  sleep 5
done
