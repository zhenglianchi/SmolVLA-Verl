#!/bin/bash
# 每10分钟追加一条训练状态到 monitor_training.log
LOG=/home/ubuntu/runs/monitor_training.log
LOOP=/home/ubuntu/grpo_loop.log
SERVE=/home/ubuntu/runs/serve.log
mkdir -p /home/ubuntu/runs
{
  echo "=== $(date '+%F %T') ==="
  echo "round: $(grep -a 'round [0-9]*/' "$LOOP" 2>/dev/null | tail -1)"
  echo "sr:    $(grep -a 'success rate' "$LOOP" 2>/dev/null | tail -1)"
  echo "train: $(grep -a '\[train\]' "$SERVE" 2>/dev/null | tail -1)"
  echo "wts:   $(ls -la /home/ubuntu/runs/smolvla_grpo/model.safetensors 2>/dev/null | awk '{print $6, $7, $8}')"
  echo "gpu:   $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
  echo "collect: $(ps aux | grep collect_remote | grep -v grep | wc -l)"
} >> "$LOG" 2>&1