# verl-vla（本项目内置源码，用于学习）

本目录是官方 [verl-project/verl-vla](https://github.com/verl-project/verl-vla) 的完整源码快照，
固定在上游 commit `74df2cb`（`feat: add PI0.5 DSRL experiment recipes`），与服务器训练环境使用
的版本一致。

## 为什么内置在仓库里

- 训练/评估全程依赖 verl-vla 的模型注册、rollout worker 与 workflow；内置后本地可以直接
  `rg` / 跳转 / diff，学习 VLA 的 RL 训练方式更方便（参考 uniagent-lighting 的 `vendor/` 模式）。
- 我们的改动全部收敛在 `../patches/` 里，可以随时重新应用，也能和上游 diff。

## 我们加在 verl-vla 里的东西

1. `src/verl_vla/models/smolvla/` —— SmolVLA 模型包：
   - `sde.py` / `sde_sampling.py`：边缘保持 SDE 核、chunk 采样与重打分
   - `grpo.py`：GRPO 损失 + 组优势 + KL
   - `trainable_model.py`：verl-vla 可训练包装（rollout/sft/flow hooks）
   - `modeling.py` / `processor.py` / `configuration.py`：模型本体与处理器
2. `src/verl_vla/models/builder.py` —— 新增 `architecture == "smolvla"` 分支注册
3. `src/verl_vla/workers/config/model.py` —— `policy_type == "smolvla"` 映射 + tokenizer/processor 置空
4. `scripts/smoke_smolvla_build.py` —— Stage-A 冒烟：builder 能否构建出 SmolVLA 可训练模型

## 重新应用补丁

```bash
cd verl-vla
python3 ../patches/apply_patch.sh  # 幂等：已 patch 的文件会 SKIP
```
