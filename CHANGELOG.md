# Changelog

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
- M1 基线完成：LIBERO-Spatial 60.0%（官方协议 10 trials/任务）
