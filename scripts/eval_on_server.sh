#!/usr/bin/env bash
# 服务器官方评估（10 任务 x 10 trials）
# - n_action_steps=5: 与采集一致的回退式 chunk 执行，推理量 1/5，保证 14:00 前出结果
# - empty_cameras=0（不覆盖）: 与 serve 推理一致，规避 misaligned address
set -euo pipefail
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV=/home/ubuntu/vla_serve
POLICY=${POLICY:-/home/ubuntu/runs/smolvla_grpo}
BASE=${BASE:-/home/ubuntu/models/smolvla_libero}
TASK=${TASK:-libero_spatial}
OUT=${OUT:-/home/ubuntu/results/grpo_${TASK}}
mkdir -p /home/ubuntu/results
# copy processors (unchanged by training) from the base checkpoint
cp -f "$BASE/policy_preprocessor.json" "$POLICY/" 2>/dev/null || true
cp -f "$BASE"/policy_preprocessor_step_*.safetensors "$POLICY/" 2>/dev/null || true
cp -f "$BASE/policy_postprocessor.json" "$POLICY/" 2>/dev/null || true
cp -f "$BASE"/policy_postprocessor_step_*.safetensors "$POLICY/" 2>/dev/null || true
echo "[eval] processors copied from $BASE to $POLICY"
ls "$POLICY"
"$ENV/bin/python" -m lerobot.scripts.lerobot_eval \
  --policy.path="$POLICY" \
  --env.type=libero --env.task="$TASK" \
  --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false \
  --policy.device=cuda --policy.n_action_steps=5 \
  --env.camera_name_mapping='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}' \
  --rename_map='{"observation.images.camera1":"observation.images.image","observation.images.camera2":"observation.images.image2"}' \
  --seed=1000 \
  --output_dir="$OUT"
echo "[eval] done, parsing results"
"$ENV/bin/python" /home/ubuntu/SmolVLA-Verl/scripts/parse_eval_results.py "$OUT/eval_info.json" | tee /home/ubuntu/results/grpo_${TASK}_summary.txt
echo "[eval] EVAL_OK"