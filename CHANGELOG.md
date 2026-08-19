# Changelog

## v0.3.0 (2026-08-19)
- 仓库重构为学习友好布局（参考 uniagent-lighting）：官方 **verl-vla 完整源码内置到顶层**
  （固定上游 commit `74df2cb` + 已应用 SmolVLA 注册补丁），`verl_vla_changes/` 更名为 `patches/`。
- 文档重写：README 学习路径、`docs/learning-verl-vla.md`（verl-vla 导读）、架构/评测/安装更新、
  RESULTS 修正为真实数字（GRPO 15 轮 63.0% = Base 63.0%，修复后无退化）。
- 清理：移除 `serve_smolvla.py` 的 `/dump` pickle 调试端点（调试技巧写入文档）；清理本地/服务器
  临时日志与临时评估目录；保留正式训练日志 `work/logs/grpo_opt.log`、配置与最终评估结果。
- `configs/grpo_formal.yaml` 更新为正式 run 实际参数（12 实例 × 4、lr=5e-6、steps=1、batch=32）。

## v0.2.0 (2026-08-18)
- 修复 GRPO 组不同初始场景：采集端固定 `LiberoEnv.init_state_id`，组内 4 条 episode 同场景（reset-matched），并按轮次轮换 init state 0-9 覆盖评估分布。
- 修复终局 chunk 的 `valid_positions`：客户端回传每 chunk 实际执行步数，服务端掩掉未执行的规划后缀动作。
- 训练目标修复：只更新 mixed 组（跳过全成功/全失败组）；chunk 按 episode 均衡加权（失败长 episode 不再主导梯度）；新增 reward-to-go `chunk_discount`（默认 0.99）把 credit 集中到终局 chunk。
- 数值一致性守卫：`/train` 与离线训练首个前向 ratio 偏离 1 超过 0.05 直接报错；会话图像改为 float32 存储。
- `grpo_libero.py` 修复：使用环境真实任务指令（去掉硬编码 TASK_DESC）、任务轮换覆盖全部 task、rollout worker 每轮重载最新权重（on-policy）、`pool.map` 整轮并行、组内同场景。
- 训练 SDE 噪声 eta 默认 0.1 → 0.05，减小与确定性 ODE 评估的口径差异。
- 修复离线训练器 `grpo_offline.py` 中 `preprocessor/postprocessor` 未定义的 NameError。

## v0.1.0 (2026-08-16)
- 初始平台化仓库：FlowGRPO 训练 + 评估 + 采集上传脚本
- M1 基线完成：LIBERO-Spatial 60.0%（官方协议 10 trials/任务，后重测修正为 63.0%）
