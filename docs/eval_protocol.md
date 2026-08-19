# 评测协议（与论文官方一致）

- 每任务 **10 trials**，LIBERO 每套件 10 任务 → 每套 100 个 episode
- eval seed 1000；step caps：Spatial/Object 280、Goal 300、Long 520
- 确定性 ODE 采样评估（不是 SDE）；二进制成功判定（任务完全完成 = 1）
- 推理时 `n_action_steps=5`（与采集端一致的 chunk 回退执行），单进程逐任务评估容易触发偶发
  CUDA 崩溃，因此正式评估用**逐任务独立进程并行**（`eval_parallel.sh`，两波 × 5 任务，失败可重跑单任务）

## 基线（实测）

| 套件 | 官方 SmolVLA 0.45B | 本项目实测（lerobot eval，seed=1000） |
|---|---|---|
| LIBERO-Spatial | 90% | **63.0%**（2026-08-19 重测两次一致；2026-08-17 首次 63.0%） |

> 社区复评普遍比论文低（协议/实现差异），63.0% 是可信可复现基线。GRPO 修复后官方指标 63.0% 与
> base 持平，未退化也未超越；100 集评估噪声约 ±5%。

## 评估命令

```bash
# GRPO 权重
OUT=/home/ubuntu/results/grpo_final_pertask POLICY=/home/ubuntu/runs/smolvla_grpo \
  bash scripts/eval_parallel.sh
# Base 权重
OUT=/home/ubuntu/results/base_final_pertask POLICY=/home/ubuntu/models/smolvla_libero \
  bash scripts/eval_parallel.sh
# 解析
python3 scripts/parse_eval_results.py --dir "$OUT"
```
