# results/（评估结果索引）

完整数据与逐任务表格见根目录 `RESULTS.md`。服务器上保留原始评估产物：

- **本仓库**：`grpo_final_pertask_run2/` 与 `base_final_pertask_run2/` —— 最终评估逐任务原始日志
  （`task_N.log` + `task_N/eval_info.json`，无视频），63.0% 的可追溯证据
- 服务器 `/home/ubuntu/results/grpo_final_pertask_run2/`、`base_final_pertask_run2/` —— 同一份 +
  100 集评估视频（服务器关闭前如需保留视频请自行备份）
- 早期基线/GRPO summary 文本也保留在仓库：`base_libero_spatial_pertask_summary.txt`、
  `grpo_libero_spatial_pertask_summary.txt`

历史上临时评估目录（probe、smoke、R5、recheck 等）已清理，汇总数字保留在 `RESULTS.md`。
