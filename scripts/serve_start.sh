#!/bin/bash
# 启动/重启 serve（加载最新权重），带正确 PYTHONPATH + 离线 HF 缓存；CHECKPOINT 可覆盖
cd /home/ubuntu/SmolVLA-Verl
export PYTHONPATH=/home/ubuntu/SmolVLA-Verl/src:/home/ubuntu/SmolVLA-Verl/src/smolvla_verl/models/smolvla
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
CHECKPOINT=${CHECKPOINT:-/home/ubuntu/runs/smolvla_grpo}
pkill -9 -f serve_smolvla 2>/dev/null
sleep 3
setsid nohup /home/ubuntu/vla_serve/bin/python scripts/serve_smolvla.py \
  --checkpoint "$CHECKPOINT" \
  --save-dir /home/ubuntu/runs/smolvla_grpo \
  --chunk-size 50 --action-steps 5 --port 8000 --host 0.0.0.0 \
  >> /home/ubuntu/runs/serve.log 2>&1 < /dev/null &
echo "SERVE_PID=$!"
for h in $(seq 1 90); do
  if curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"'; then
    echo "SERVE_HEALTHY after ${h} tries"
    exit 0
  fi
  sleep 2
done
echo "SERVE_NOT_HEALTHY"
tail -n 5 /home/ubuntu/runs/serve.log | tr -d "\r" | tail -3
exit 1