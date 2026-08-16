#!/usr/bin/env bash
# 把 SmolVLA 支持应用到官方 verl-vla（在 verl-vla 仓库根目录运行，或指定 VLA_REPO）
set -euo pipefail
REPO=${VLA_REPO:-$(pwd)}
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../src/smolvla_verl/models/smolvla"

echo ">> copying models/smolvla -> $REPO/src/verl_vla/models/smolvla"
mkdir -p "$REPO/src/verl_vla/models/smolvla"
cp -r "$SRC/." "$REPO/src/verl_vla/models/smolvla/"

echo ">> patching builder.py + workers/config/model.py"
"${PYTHON:-python3}" "$HERE/patch_registration.py" "$REPO"
echo ">> done"
