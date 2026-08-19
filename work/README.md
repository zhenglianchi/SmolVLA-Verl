# work/（运行产物索引）

本目录只放**正式训练日志**与运行说明；大体积产物（权重、轨迹视频、逐轮采集日志）保留在服务器，
不进入 git。

| 内容 | 位置 |
|---|---|
| 正式训练日志（15 轮 GRPO） | `work/logs/grpo_opt.log`（本仓库） / 服务器 `/home/ubuntu/grpo_opt.log` |
| 逐轮逐实例采集日志 | 服务器 `/home/ubuntu/SmolVLA-Verl/work/logs/opt_r*.log`（保留最后一轮） |
| 最终权重 | 服务器 `/home/ubuntu/runs/smolvla_grpo`（15 轮最终）、`smolvla_best`（最佳）、`smolvla_grpo_r16_bak`（旧管道） |
| 正式评估结果（含视频） | 服务器 `/home/ubuntu/results/grpo_final_pertask_run2`、`base_final_pertask_run2` |

> 说明：在线训练的轨迹 session 保存在 serve 进程内存中，每轮 `/train` 消费后 `/clear`，没有落盘
> 完整轨迹；逐轮 episode 成功/步数记录在采集日志中。如需保存完整轨迹，可临时在
> `serve_smolvla.py` 加 pickle dump（调试技巧见 `docs/learning-verl-vla.md`）。
