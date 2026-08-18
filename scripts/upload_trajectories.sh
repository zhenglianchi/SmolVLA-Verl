#!/usr/bin/env bash
# 上传本地采集的轨迹到服务器（SFTP/scp）
set -euo pipefail
SRC=${1:-work/trajectories/traj.pkl}
HOST=${SERVER_HOST:-117.50.189.92}
USER=${SERVER_USER:-ubuntu}
DEST=${SERVER_TRAJ_DIR:-/home/ubuntu/trajectories}
export SSHPASS="${SERVER_PASS:-zlc131310}"
sshpass -e ssh -o StrictHostKeyChecking=no "$USER@$HOST" "mkdir -p $DEST"
sshpass -e scp -o StrictHostKeyChecking=no "$SRC" "$USER@$HOST:$DEST/"
echo "uploaded $SRC -> $HOST:$DEST/"
