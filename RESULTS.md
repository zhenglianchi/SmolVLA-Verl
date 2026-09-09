# 实验结果（全部为真实测量）

所有数字均来自服务器 `/home/ubuntu/results/` 下的逐任务评估日志，协议为 LIBERO-Spatial 官方
10 任务 × 10 trials（seed=1000，确定性 ODE 采样，`n_action_steps=5` 与采集端一致）。

## 评测协议与基线（官方口径）

- 每任务 **10 trials**，LIBERO 每套件 10 任务 → 每套 100 个 episode
- eval seed 1000；step caps：Spatial/Object 280、Goal 300、Long 520
- 确定性 ODE 采样评估（不是 SDE）；二进制成功判定（任务完全完成 = 1）
- 推理时 `n_action_steps=5`（与采集端一致的 chunk 回退执行）；单进程逐任务评估容易触发偶发
  CUDA 崩溃，正式评估用**逐任务独立进程并行**（`eval_parallel.sh`，两波 × 5 任务，失败可重跑单任务）

| 套件 | 官方 SmolVLA 0.45B | 本项目实测（lerobot eval，seed=1000） |
|---|---|---|
| LIBERO-Spatial | 90% | **63.0%**（2026-08-19 重测两次一致；2026-08-17 首次 63.0%） |

> 社区复评普遍比论文低（协议/实现差异），63.0% 是可信可复现基线。GRPO 修复后官方指标 63.0% 与
> base 持平，未退化也未超越；100 集评估噪声约 ±5%。

## 一、最终评估（2026-08-19，修复后 15 轮 GRPO）

**GRPO 15 轮**（`results/grpo_final_pertask_run2`，与 run1 完全一致，确定性复现）：

| task | GRPO-15轮 | Base（同日重测 `base_final_pertask_run2`） |
|------|-----------|------|
| 0 | 8/10 | 5/10 |
| 1 | 5/10 | 8/10 |
| 2 | 8/10 | 7/10 |
| 3 | 4/10 | 4/10 |
| 4 | 6/10 | 4/10 |
| 5 | 2/10 | 6/10 |
| 6 | 8/10 | 8/10 |
| 7 | 5/10 | 6/10 |
| 8 | 9/10 | 7/10 |
| 9 | 8/10 | 8/10 |
| **总体** | **63/100 = 63.0%** | **63/100 = 63.0%** |

- Base 重测两次结果一致（63.0%），GRPO 重测两次结果一致（63.0%），均为 seed=1000 确定性。
- 修复前旧管道（R16，2026-08-18）：**55.0%**，本次修复后无退化。

### 训练过程采集成功率（15 轮 × 48 集，任务/初始状态轮换）

| Round | 成功率 | Round | 成功率 |
|-------|--------|-------|--------|
| 1 | 62.5% | 9 | 72.9% |
| 2 | 70.8% | 10 | 70.8% |
| 3 | 56.2% | 11 | 64.6% |
| 4 | 60.4% | 12 | 56.2% |
| 5 | 45.8% | 13 | 66.7% |
| 6 | 66.7% | 14 | 47.9% |
| 7 | 68.8% | 15 | 62.5% |
| 8 | 56.2% | **均值** | **~62%** |

> 采集成功率只反映在线 rollout 表现（初始状态与任务逐轮轮换），不代表官方指标；训练全程无崩溃、
> 无退化（watchdog 未触发），ratio_mean 全程稳定在 1.000000 附近。

### 配置（正式 run；`configs/grpo_formal.yaml` 仅为备忘不加载，实际参数由 `run_loop_opt.sh` env 控制，
正式日志见 `work/logs/grpo_opt.log`）

- 12 实例 × 4 rollout = 48 集/轮，15 轮，任务轮换 + 初始状态轮换 `init_state=(r+i)%10`
- verl 式固定 base 锚定：训练时重算 old log-prob（逐 chunk 重打分），`lr=5e-6, steps=1, batch=32`
- `chunk_discount=0.99`（reward-to-go 把 credit 集中到终局 chunk），`eta=0.05`（SDE 噪声）
- 单权重夹覆盖：`/home/ubuntu/runs/smolvla_grpo` + 最佳权重备份 `smolvla_best`

## 二、历史评估

### 2026-08-18：旧管道 R16（verl 式固定 base 锚定，修复前）

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 总体 |
|------|---|---|---|---|---|---|---|---|---|---|------|
| Base | 70 | 80 | 90 | 50 | 40 | 30 | 80 | 50 | 70 | 70 | 63.0% |
| GRPO-R16 | 50 | 80 | 50 | 70 | 40 | 40 | 70 | 20 | 80 | 50 | 55.0% |

### 2026-08-17：早期 R3 / R5（未锚定）

- GRPO R3（16 集/轮，5 轮）：**62.0%**（70/90/60/40/40/40/80/60/80/60）
- GRPO R5：8/10 任务完成 63.8%（关机电量中断），与 R3 同口径
- 同期 Base 63.0%（`base_libero_spatial_pertask`）

## 三、发现并修复的问题（正确性）

1. **批处理重打分破坏 log-prob**：`/train` 对整批会话一次重打分，`valid_positions` 与分组不一致导致
   ratio 漂移；修复为逐 chunk 重打分，并与采集时 log-prob 做首 chunk 一致性校验，偏离 >0.05 直接报错。
2. **组内不同初始场景**：采集端未固定 `init_state_id`，同组 4 条 rollout 场景不一致，组均值基线失效；
   修复为 reset-matched 同场景，并按轮次轮换 init state 0-9 覆盖评估分布。
3. **终局 chunk mask 错误**：客户端回传每 chunk 实际执行步数，服务端掩掉未执行的规划后缀动作。
4. **训练目标只应更新 mixed 组**：全成功/全失败组优势无信息，跳过；chunk 按 episode 均衡加权，
   失败长 episode 不再主导梯度。
5. **`grpo_loss` 加权分母 bug**：分母被 `clamp(min=1)` 导致 loss 被错误缩放；修复为按 mask 求和。
6. **`final_info` 解析**：可能是 dict 或 list，统一解析 episode 是否成功。

## 四、根因分析（为什么 GRPO 没有超越 base）

1. **初始状态分布不匹配**：评估测 10 个初始状态（0-9），早期训练只覆盖部分（过拟合训练初始状态），
   修复后按轮轮换已缓解，但每轮每个初始状态只出现 ~1 次，采样仍稀疏。
2. **数据量偏少**：48 集/轮 × 15 轮，且其中约 1/3 全成功/全失败组不产生梯度。
3. **紧 KL 锚 + 低 lr**（防退化优先）：单轮学习幅度小。
4. **评估噪声**：100 集 ±5%；多次运行 62-64% 均在 base 附近波动，该数据量下提升不显著。

## 五、复现路径

```bash
# 训练（15 轮示例）
STOP_AT="2026-08-19 17:00" ROUNDS=15 bash scripts/run_loop_opt.sh
# GRPO 评估（与正式 run 相同）
OUT=/home/ubuntu/results/grpo_final_pertask POLICY=/home/ubuntu/runs/smolvla_grpo \
  bash scripts/eval_parallel.sh
# Base 评估
OUT=/home/ubuntu/results/base_final_pertask POLICY=/home/ubuntu/models/smolvla_libero \
  bash scripts/eval_parallel.sh
```

服务器保留产物：`/home/ubuntu/runs/smolvla_grpo`（最终权重）、`smolvla_best`（最佳）、
`smolvla_grpo_r16_bak`（旧管道权重）、`/home/ubuntu/results/grpo_final_pertask_run2` 与
`base_final_pertask_run2`（正式评估含视频）、`work/logs/grpo_opt.log`（正式训练日志）。

## 六、训练配置演化（M1 → M3）

### M1（2026-08-16~17）：第一版 FlowGRPO

- 4 实例 × 4 rollout = 16 集/轮，`lr=1e-6`、SDE eta 偏大、无固定 base 锚定
- 结果：R3 官方 62.0%（与 base 63.0% 接近），R5 关机电量中断
- 问题：无 KL 锚定 → 策略漂移，后期采集成功率降到 45%

### M2（2026-08-18）：verl 式固定 base 锚定（旧管道 R16）

- 12 实例 × 4 = 48 集/轮，16 轮；固定 base 参考做 KL 锚定；`lr=5e-6, steps=3, batch=32`
- 采集成功率全程稳定（~65%），**但官方指标 55.0% < Base 63.0%，训练后反而退化**
- 诊断：批量重打分破坏 logp、组内场景不一致、终局 mask 错误、加权分母 bug

### M3（2026-08-19）：修复后 15 轮（正式结果）

- 逐 chunk 重打分 + ratio 守卫、reset-matched 组、终局 mask、mixed 组才更新、
  episode 均衡加权、`chunk_discount=0.99`、eta=0.05、`steps=1`
- **GRPO 63.0% = Base 63.0%**：修复了退化，但未超越 base（重测 ×2 一致）

## 七、如果继续做（下一步假设）

1. 采集初始状态 0-9 全覆盖且每状态多采样（当前最大嫌疑）
2. 每组 rollout 4 → 8，放宽 KL / 提高 lr（监控 ratio_mean 稳定在 1 附近再放）
3. 评估 200+ 集或多次运行取均值，区分信号与噪声
