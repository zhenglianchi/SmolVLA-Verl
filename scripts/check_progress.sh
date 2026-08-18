#!/usr/bin/env bash
# 查看训练进度：轮次/成功率/训练指标/当前权重时间戳
LOG=${1:-/home/ubuntu/grpo_loop.log}
echo "=== 循环日志 ==="
tail -20 "$LOG"
echo "=== 服务器训练指标 ==="
grep -a "\[train\]" /home/ubuntu/runs/serve.log | tail -5
echo "=== 当前权重 ==="
ls -la /home/ubuntu/runs/smolvla_grpo/model.safetensors
echo "=== 循环进程 ==="
ps aux | grep run_grpo_loop | grep -v grep | awk "{print \$2, \$3}" | head -1
