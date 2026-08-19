# verl-vla 学习导读

这份导读面向「想搞懂 VLA 的 RL 训练是怎么跑起来的人」。仓库顶层 `verl-vla/` 就是官方源码
（固定 commit `74df2cb`），可以直接 `rg`、跳转、改代码。

## 1. verl-vla 是什么

verl-vla 是 [verl](https://github.com/verl-project/verl) 之上的 **VLA post-training 框架**：
把「数据采集 → 微调 → 强化学习」统一到一套执行架构里。模型、环境、训练算法各自独立接入，
可以自由组合。核心抽象：

- **模型**：每个架构一个包（`src/verl_vla/models/<arch>/`），统一暴露 `build` + `trainable model`
- **worker**：rollout worker（采样）、train worker（更新）、env worker（仿真/真机）
- **workflow / entrypoint**：`record`（采集）、`sft`（微调）、`ppo`/`sac`（RL）、`eval`（评估）

## 2. 目录地图

| 路径 | 内容 |
|---|---|
| `src/verl_vla/models/` | 模型包：`act_torch`、`gaussian_actor`、`openvla_oft`、`pi0_torch`、`gr00t`、**`smolvla`（本项目加的）**；`builder.py` 统一注册 |
| `src/verl_vla/workers/` | 执行 worker：`config/`（超参 schema）、`env/`、`rollout/`、`engine/` |
| `src/verl_vla/trainer/` | 训练算法：`sft/`、`ppo/`、`sac/` |
| `src/verl_vla/workflows/` | 流程编排：`train/`、`eval.py`、`record.py`、`replay.py`、`dagger.py`、`teleop.py` |
| `src/verl_vla/entrypoints/` | CLI 入口（调用对应 workflow） |
| `examples/rl/` | 可复现 RL 实验配置：`recap/`（PI0.5）、`td3_bc/`、`sac/`、`dsrl/` |
| `docs/reinforcement-learning/` | 官方 RL 文档（recap / td3-bc / dsrl） |
| `docs/fine-tuning/` / `docs/data-collection/` | 微调与采集文档 |

## 3. 官方 RL 链路长什么样（以 RECAP 为例）

```text
examples/rl/recap/pi05/.../libero10_task8.yaml   ← 实验配置
        │
        ▼
verl_vla/entrypoints/train.py  →  workflows/train/<algo>_workflow.py
        │
        ├── worker: rollout（仿真环境 + 策略采样 + reward 计算）
        ├── worker: train（PPO 等算法更新）
        └── 循环：rollout 数据 → 训练 → 同步权重 → 下一轮
```

读代码顺序建议：

1. `examples/rl/recap/pi05/libero10_task8_from_sft_10demos/libero10_task8.yaml` —— 一个 RL 实验需要哪些配置
2. `src/verl_vla/workers/config/` —— 配置 schema（model / rollout / train 三段）
3. `src/verl_vla/models/builder.py` —— 模型注册入口（本项目 patch 加的就是 `architecture == "smolvla"` 分支）
4. `src/verl_vla/trainer/ppo/` —— 算法实现（本项目用的是 GRPO 变体，见下）
5. `src/verl_vla/workflows/train/` —— 训练循环怎么把 rollout 和 train 串起来

## 4. 本项目怎么接入（FlowGRPO）

本项目没有走 verl-vla 的多卡 trainer，而是单机 HTTP serve + 离线 GRPO，但复用了它的模型基建：

```text
patches/apply_patch.sh
  ├── 拷贝 src/smolvla_verl/models/smolvla/ → verl-vla/src/verl_vla/models/smolvla/
  └── patch builder.py + workers/config/model.py（注册 smolvla 架构）

src/smolvla_verl/trainer/grpo_libero.py   ← 训练主循环（采集→训练→覆盖权重）
scripts/serve_smolvla.py                  ← predict/finish/train/clear
src/smolvla_verl/trainer/grpo_offline.py  ← 从记录的 session 做 GRPO 更新
```

核心算法文件都在 `verl-vla/src/verl_vla/models/smolvla/`：

| 文件 | 学什么 |
|---|---|
| `sde.py` | flow matching 的 SDE 化：为什么 ODE 没有 logp、SDE 怎么有闭式 log-density |
| `sde_sampling.py` | 带 prefix cache 的 chunk 采样、逐 chunk 重打分（与采集共用数值路径） |
| `grpo.py` | GRPO 损失：组均值基线、逐去噪步 ratio、参考策略 KL、episode 均衡加权 |
| `trainable_model.py` | 模型如何暴露给 verl-vla（rollout/sft/flow hooks） |

## 5. 关键概念速记

- **Flow matching**：动作不是自回归生成的，而是从噪声按 ODE 去噪 10 步解出整个动作 chunk。
- **SDE 边缘保持**：把确定性去噪换成带噪声的随机转移，噪声边缘分布不变，但每个转移有了
  `log p(x_{t-h}|x_t)` 的闭式解 —— 这是能算策略梯度（GRPO）的前提。
- **GRPO**：同一初始场景采样 G 条 rollout，组内成功率的均值做基线，减去基线后的二进制奖励
  除以组标准差，再乘 clipped ratio + KL 惩罚。全成功/全失败组没有信息量，跳过。
- **KL 锚定**：固定 base 参考策略算 KL，防止策略漂移太远（代价是学习幅度小）。
- **on-policy**：每轮必须用最新权重重新采集，否则 log-prob 对不上，训练退化。

## 6. 本次踩过的坑（调试时先查这里）

1. 批处理重打分会让 `valid_positions` 错位 → 必须逐 chunk 重打分，并校验 ratio ≈ 1（偏差 >0.05 报错）。
2. 同组 rollout 必须 reset-matched 同初始场景，否则组均值基线没意义。
3. 终局 chunk 要按实际执行步数 mask 掉未执行的动作。
4. 采样 SDE 噪声 `eta` 与评估的确定性 ODE 口径要尽量一致（0.05）。
5. 每轮训练后必须重启 serve 让 rollout worker 加载最新权重。

## 7. 推荐的后续实验

- 采集初始状态随机化（0-9 全覆盖且每状态多采样）——当前最大嫌疑
- 每组 rollout 4 → 8，放宽 KL / 提高 lr（先看 ratio_mean 是否稳定在 1 附近）
- 评估 100 集噪声 ±5%，想看到统计显著差异需要 200+ 集或多次运行取均值
