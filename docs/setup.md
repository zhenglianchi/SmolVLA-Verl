# 安装（本地 WSL2 与服务器一致）

## 依赖

- Python 3.12，Ubuntu 24.04
- 训练侧：1×24GB GPU（RTX 4090）；本地 8GB 只做采集/推理
- pip 用清华源；HF 用 hf-mirror.com

## 步骤（服务器 /home/ubuntu）

```bash
# 系统依赖
sudo apt-get install -y build-essential cmake ffmpeg libgl1 libglib2.0-0 libosmesa6 python3.12-dev python3.12-venv
# venv + 依赖
python3.12 -m venv /home/ubuntu/vla_verl
source /home/ubuntu/vla_verl/bin/activate
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip setuptools wheel
pip install -r verl-vla/requirements-lerobot.txt        # torch 2.7.1 等
pip install --no-deps lerobot==0.4.4
pip install verl==0.7.1
pip install fastapi "uvicorn[standard]" scipy num2words
cd verl-vla && pip install -e ".[libero]"
# LIBERO assets（走 hf-mirror 的 lerobot/libero-assets，或 GitHub codeload 需代理）
python -c "from lerobot.envs.libero import _get_suite"  # 触发 assets 下载（如缺失）
```

> 仓库内置的 `verl-vla/` 已经应用了 SmolVLA 注册补丁（`patches/`），不需要再手动 patch；
> 若从上游重新同步，跑 `python3 patches/apply_patch.sh`。

## 权重

```bash
# 从 hf-mirror 下载 SmolVLA-LIBERO checkpoint
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
huggingface-cli download lerobot/smolvla_libero --local-dir /home/ubuntu/models/smolvla_libero
# SmolVLM2-500M-Instruct 首次加载时自动从 hf-mirror 拉取
```

## 冒烟

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
# Stage-A：verl-vla builder 能否构建 SmolVLA 可训练模型
python3 verl-vla/scripts/smoke_smolvla_build.py
# Stage-B：单任务 1 轮 GRPO（2 集）
bash scripts/run_grpo_smoke.sh
```

## 三端同步约定

- 代码仓：`github.com/zhenglianchi/SmolVLA-Verl` 为准
- 权重不拉回本地，保留在服务器 `/home/ubuntu/runs/`
- 每次调试完成后：本地 `git push` → 服务器 `git pull`（或 rsync），保持一致
