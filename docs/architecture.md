# 架构

```
本地（8GB 4060 / WSL2）            服务器（RTX 4090 24GB）
┌─────────────────────┐           ┌──────────────────────────────┐
│ LIBERO 轨迹采集      │  上传      │ verl-vla（含 smolvla 扩展）    │
│ scripts/collect_*   │ ────────▶ │ src/smolvla_verl/trainer/    │
│ trajectory_uploader │  SFTP/scp │   grpo_libero.py 训练主循环    │
└─────────────────────┘           │   - 12 实例并行 rollout       │
        ▲                         │   - FlowGRPO（SDE logp + GRPO）│
        │ 拉权重/评测结果          │   - 15 轮、续训、单权重夹覆盖   │
        └─────────────────────────├──────────────────────────────┤
                                  │ run_eval → 官方协议评估        │
                                  └──────────────────────────────┘
```

## 算法（FlowGRPO）

1. SmolVLA 推理 = 确定性 ODE 流匹配采样（10 步去噪，50 步 action chunk），本身没有 log-prob。
2. 改造为**边缘保持 SDE**：每步去噪 = 高斯转移 `p(x_{t-h}|x_t)`，闭式 log-density；各噪声层边缘分布与 ODE 相同
   （`verl-vla/src/verl_vla/models/smolvla/sde.py`）。
3. GRPO：同初始场景 G 条 rollout（组均值基线）、LIBERO 二进制成功奖励、逐去噪步裁剪 ratio + 参考策略 KL
   （`verl-vla/src/verl_vla/models/smolvla/grpo.py`）。
4. 关键：采集与重打分**必须共用同一数值路径**（bf16 autocast、逐 chunk 重打分），否则 ratio 偏离 1 训练退化。

## 训练数据流（在线）

```
run_loop_opt.sh
  └─ 每轮起 12 个 collect_remote.py（任务/初始状态轮换，4 rollout/实例）
        └─ POST /predict → serve_smolvla.py 记录每 chunk（obs + SDE 状态 + logp）
        └─ POST /finish  → 组优势记账（同场景 reset-matched）
  └─ POST /train       → grpo_offline.train_from_sessions
        └─ 逐 chunk 重打分（校验与采集 logp 一致）→ GRPO 更新 → 保存权重
  └─ POST /clear + 重启 serve → 下一轮用最新权重（on-policy）
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `verl-vla/src/verl_vla/models/smolvla/sde.py` | 边缘保持 SDE 核（score/transition/logprob） |
| `verl-vla/src/verl_vla/models/smolvla/sde_sampling.py` | prefix cache + chunk 采样 + 重打分 |
| `verl-vla/src/verl_vla/models/smolvla/grpo.py` | GRPO 损失 + 组优势 + k3 KL |
| `verl-vla/src/verl_vla/models/smolvla/trainable_model.py` | verl-vla 可训练包装（rollout/sft/flow hooks） |
| `src/smolvla_verl/trainer/grpo_libero.py` | 训练主循环（多 runner/续训/单权重夹） |
| `src/smolvla_verl/trainer/grpo_offline.py` | `/train` 使用的离线训练器（从记录的 session 学习） |
| `scripts/serve_smolvla.py` | 服务端：predict/finish/train/clear/stats |
| `scripts/run_loop_opt.sh` | 正式训练编排（48 集/轮 + 时间截止 + 断点续训） |
| `scripts/eval_parallel.sh` | 官方评估（10 任务并行，逐任务独立进程，seed=1000） |

## 为什么不用 verl-vla 自带的 trainer？

verl-vla 的完整分布式 trainer（Ray + 多 worker）适合多卡集群；本项目单卡 4090 + 12 个采集进程，
直接复用 verl-vla 的**模型注册/构建/处理器**基建，而训练循环用轻量 HTTP serve + 离线 GRPO 实现，
便于单机调试、断点续训和权重夹覆盖。两者在 `trainable_model.py` 的 hook 接口上兼容。
