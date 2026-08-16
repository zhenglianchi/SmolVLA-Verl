# 架构

```
本地（8GB 4060 / WSL2）            服务器（RTX 4090 24GB）
┌─────────────────────┐           ┌──────────────────────────────┐
│ LIBERO 轨迹采集      │  上传      │ verl-vla（含 smolvla 扩展）    │
│ scripts/collect_*   │ ────────▶ │ trainer/grpo_libero.py        │
│ trajectory_uploader │  SFTP/scp │   - rollout workers (N 进程)  │
└─────────────────────┘           │   - FlowGRPO（SDE logp + GRPO）│
        ▲                         │   - 10 轮、续训、单权重夹覆盖   │
        │ 拉权重/评测结果          ├──────────────────────────────┤
        └─────────────────────────│ run_eval.sh → 官方协议评估     │
                                  └──────────────────────────────┘
```

## 算法（FlowGRPO）
1. SmolVLA 推理 = 确定性 ODE 流匹配采样（10 步去噪，50 步 action chunk），无 log-prob。
2. 改造为**边缘保持 SDE**：每步去噪 = 高斯转移 `p(x_{t-h}|x_t)`，闭式 log-density；各噪声层边缘分布与 ODE 相同（`models/smolvla/sde.py`）。
3. GRPO：同初始场景 G 条 rollout（组均值基线）、LIBERO 二进制成功奖励、逐去噪步裁剪 ratio + 参考策略 KL（`models/smolvla/grpo.py`）。
4. 关键：采集与重打分**必须共用同一数值路径**（bf16 autocast），否则 ratio 偏离 1 训练退化。

## 关键文件
| 文件 | 作用 |
|---|---|
| `src/smolvla_verl/models/smolvla/sde.py` | 边缘保持 SDE 核（score/transition/logprob） |
| `src/smolvla_verl/models/smolvla/sde_sampling.py` | prefix cache + chunk 采样 + 重打分 |
| `src/smolvla_verl/models/smolvla/grpo.py` | GRPO 损失 + 组优势 + k3 KL |
| `src/smolvla_verl/models/smolvla/trainable_model.py` | verl-vla 可训练包装（rollout/sft/flow hooks） |
| `src/smolvla_verl/trainer/grpo_libero.py` | 训练主循环（多 runner/续训/单权重夹） |
