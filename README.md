# SmolVLA-Verl

基于 [verl-vla](https://github.com/verl-project/verl-vla) 的 **SmolVLA 流匹配 VLA 在线强化学习（FlowGRPO）训推平台**，LIBERO-Spatial 全链路闭环：

本地/边缘轨迹采集 → 服务器（RTX 4090 24GB）GRPO 训练 → 单权重夹覆盖 → 官方协议自动评估。

## 实验结果（真实数据，LIBERO-Spatial，官方协议 10 任务 × 10 trials）

| 模型 | 成功率 |
|------|--------|
| Base（SmolVLA-LIBERO 预训练，今天重测 ×2） | **63.0%** |
| GRPO（修复后 15 轮 × 48 集，今天评估 ×2） | **63.0%** |
| GRPO（旧管道 R16，2026-08-18） | 55.0% |

**结论（诚实）**：修复掉批处理重打分、组内不同场景、终局 mask、加权分母等正确性 bug 后，训练**不再退化**（旧管道 55% → 修复后 63%），但 GRPO 官方指标与 base 持平，未超越。

**根因**：采集初始状态分布与评估不一致（训练只覆盖部分 init_state，评估测 0-9 全分布）+ 每轮 48 集数据量偏少 + 100 集评估噪声 ±5%。详见 [RESULTS.md](RESULTS.md)。

## 目录结构（学习用，参考 uniagent-lighting 布局）

| 路径 | 说明 |
|---|---|
| `verl-vla/` | **官方 verl-vla 完整源码**（固定上游 commit `74df2cb`，已应用本项目的 SmolVLA 注册补丁），顶层内置便于学习 |
| `patches/` | 对 verl-vla 的改动（`apply_patch.sh` + `patch_registration.py`，可重新应用） |
| `src/smolvla_verl/` | 本项目核心：`models/smolvla/`（SDE 采样/重打分/GRPO 损失/可训练包装）、`trainer/grpo_libero.py`（在线训练主循环）、`trainer/grpo_offline.py`（离线训练） |
| `scripts/` | **最终链路**：采集 `collect_remote.py` + 训练 `run_loop_opt.sh` + 评估 `eval_parallel.sh`（+ `parse_eval_results.py`）、运维 `serve_start.sh` / `watchdog.sh` |
| `configs/` | GRPO/评估参数**备忘 YAML**（仅参考，脚本不加载，实际以 env/CLI 参数为准） |
| `docs/` | 架构、安装、**任务剖析**（调用链/变量/JSON/时间线）、修改与补丁、简历亮点 |
| `results/` | 评估结果汇总（正式数据见服务器 `results/`） |
| `work/` | 正式训练日志（`work/logs/grpo_opt.log`）、运行产物索引 |

## 学习路径

如果你是想学习 VLA 的 RL 训练，按这个顺序看：

1. [docs/任务剖析.md](docs/任务剖析.md) —— **任务级剖析**：一条轨迹的完整调用链、变量字典、
   完整 JSON、15 轮真实训练指标；附录 A 为端到端数据流 + 关键源码定位
2. [docs/architecture.md](docs/architecture.md) —— 平台架构与 FlowGRPO 算法
3. [docs/修改与补丁汇总.md](docs/修改与补丁汇总.md) —— 全部修改/修复清单（排障素材）
4. [docs/任务剖析.md](docs/任务剖析.md) 附录 B —— verl-vla 源码地图 + 本项目怎么接入 SmolVLA
5. `src/smolvla_verl/models/smolvla/` —— 我们加的 SDE / GRPO 核心代码（源），同一份拷贝进
   `verl-vla/src/verl_vla/models/smolvla/`
6. `src/smolvla_verl/trainer/grpo_libero.py` —— 训练主循环（采集→训练→覆盖权重）
7. [RESULTS.md](RESULTS.md) —— 实验数据、评测协议与基线、训练配置演化（M1→M3）、根因分析
8. [docs/简历亮点.md](docs/简历亮点.md) —— 简历/面试怎么讲

## 快速开始（服务器）

```bash
# 1. 环境（详见 docs/setup.md）
python3.12 -m venv /home/ubuntu/vla_verl
source /home/ubuntu/vla_verl/bin/activate
pip install -r verl-vla/requirements-lerobot.txt
pip install --no-deps lerobot==0.4.4
pip install verl==0.7.1
cd verl-vla && pip install -e ".[libero]"

# 2. 正式训练（12 实例 × ROLLOUT_N(默认 4) = 48 集/轮、lr=5e-6、steps=1、batch=32、续训、单权重夹）
#    默认跑到 60 轮，可用 ROUNDS/STOP_AT 控制，例如 15 轮：
STOP_AT="2026-08-19 17:00" ROUNDS=15 bash scripts/run_loop_opt.sh

# 3. 官方评估（10 任务并行，seed=1000 确定性）
OUT=/home/ubuntu/results/grpo_final_pertask POLICY=/home/ubuntu/runs/smolvla_grpo \
  bash scripts/eval_parallel.sh
```

## 三端同步约定

- 代码仓：`github.com/zhenglianchi/SmolVLA-Verl` 为准
- 权重不拉回本地，保留在服务器 `/home/ubuntu/runs/`
- 每次修改：本地 `git push` → 服务器 `git pull`（或 rsync）
