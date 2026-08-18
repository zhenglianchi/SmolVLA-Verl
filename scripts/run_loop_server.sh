#!/usr/bin/env bash
# 服务器 25 轮正式训练（方案A：4实例=16集/轮）+ 跑完自动评估
# 平台脚本与本地共用：collect_remote.py / run_grpo_loop.sh / serve_smolvla.py / eval_on_server.sh
set -uo pipefail
export MUJOCO_GL=egl
export SERVER=http://127.0.0.1:8000
export LOCAL_PYTHON=/home/ubuntu/vla_serve/bin/python
export ROUNDS=${ROUNDS:-25}
export INSTANCES=${INSTANCES:-4}
export CONCURRENT=${CONCURRENT:-4}
cd "$(dirname "$0")/.."
echo "=== [$(date)] 25-round training starts (4 instances, task rotation) ==="
bash scripts/run_grpo_loop.sh
echo "=== [$(date)] LOOP DONE, starting official eval (10 trials/task) ==="
bash scripts/eval_on_server.sh
echo "=== [$(date)] ALL DONE ==="
