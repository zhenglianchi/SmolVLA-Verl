# SmolVLA-Verl

基于 verl/verl-vla 的 **SmolVLA 流匹配 VLA 在线强化学习（FlowGRPO）训推平台**，LIBERO-Spatial 全链路闭环。

- 算法：把 SmolVLA 的确定性 ODE 流匹配采样改造为**边缘保持 SDE**（闭式 log-density），跑 critic-free GRPO（组均值基线 + 二进制成功奖励 + KL 惩罚）。参考 FlowVLA-RL / Flow-GRPO（arXiv:2505.05470）。
- 平台：本地/边缘只做**轨迹采集**，上传服务器训练；训练在服务器（RTX 4090 24GB）跑，多 runner 并行 rollout、续训、单权重夹覆盖；训练完自动评估。
- 基线：LIBERO-Spatial **63.0%**（10 任务 × 10 trials，官方协议，实测）。

## 目录

| 路径 | 说明 |
|---|---|
| `src/smolvla_verl/` | 核心代码：`models/smolvla/`（SDE 采样/重打分/GRPO 损失/可训练包装）、`trainer/grpo_libero.py`（训练主循环） |
| `scripts/` | 训练/评估/采集/监控/杀进程脚本 |
| `configs/` | GRPO smoke/formal、评估配置 |
| `verl_vla_changes/` | 对官方 verl-vla 的改动补丁（builder + config 注册 smolvla） |
| `docs/` | 架构、安装、评测协议、简历亮点 |
| `work/` | 配置、数据、日志、运行权重 |
| `results/` | 基线/评估结果 |

## 快速开始（服务器）

```bash
# 1. 环境（见 docs/setup.md）
python3.12 -m venv /home/ubuntu/vla_verl
source /home/ubuntu/vla_verl/bin/activate
pip install -r verl-vla/requirements-lerobot.txt
pip install --no-deps lerobot==0.4.4
pip install verl==0.7.1
cd verl-vla && pip install -e ".[libero]"
pip install num2words

# 2. 下载 checkpoint（hf-mirror）
# 3. 冒烟
bash scripts/run_grpo_smoke.sh
# 4. 正式训练（10 轮、rollout.n=4、4 runners、续训、单权重夹）
bash scripts/run_grpo_formal.sh
# 5. 评估
bash scripts/run_eval.sh
```


## 实验结果（2026-08-18，实测官方协议 10 任务 × 10 trials）

| 模型 | 成功率 |
|------|--------|
| Base（SmolVLA-LIBERO 预训练） | **63.0%** |
| GRPO（verl 式固定 base 锚定，16 轮 × 48 集） | 55.0% |

- 训练全程采集成功率稳定（~65% 均值，无退化）；GRPO 官方指标未超越 base。
- 关键发现：collect 每任务仅用 4 个初始状态（episode_index 0-3），评估测 10 个（0-9）→ 策略过拟合训练初始状态、泛化不足。
- 详见 `RESULTS.md`。