# 评测协议（与论文官方一致）

- 每任务 **10 trials**，LIBERO 每套件 10 任务 → 每套 100 个 episode
- eval seed 1000；step caps：Spatial/Object 280、Goal 300、Long 520
- 确定性 ODE 采样评估（不是 SDE）；二进制成功判定（任务完全完成 = 1）

## 基线（已实测）
| 套件 | 官方 SmolVLA 0.45B | 本项目实测（lerobot 0.6.0 eval） |
|---|---|---|
| LIBERO-Spatial | 90% | **60.0%**（100 集，~3h） |

> 注：社区复评普遍比论文低（协议/实现差异），60% 是可信可复现基线；GRPO 目标是把 60% 提上去。
