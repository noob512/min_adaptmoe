# EuroSys MoE offload evaluation workspace

本目录是 2026-08-23 之后唯一的正式论文实验入口。实验协议见
[`../../docs/eurosys_moe_offload_evaluation_plan.md`](../../docs/eurosys_moe_offload_evaluation_plan.md)。

目录约定：

- `inputs/legacy_pilot/`：为旧机制 runner 保留的确定性输入，只能用于 smoke、兼容性
  检查或生成新输入，禁止进入论文结果；
- `inputs/generated/`：按新 protocol 从 calibration split 重新生成并冻结的输入；
- `artifacts/<paper-run-id>/`：新正式运行输出，必须包含 provenance、protocol、case
  数据和顶层 verifier；
- `summaries/<paper-run-id>/`：只能从 `verification.json: status=ok` 的新 artifact
  生成。

当前 v2 E0/E1 报告见 [`RESULTS_E0_E1_20260824.md`](RESULTS_E0_E1_20260824.md)。
E0 基础兼容性已通过，E1 五重复执行已完成；但 E1 跨框架 exact-generation gate
未通过，因此其性能数字仅为描述性 anchor，尚不可进入论文 speedup 表。旧报告、汇总
和历史 artifact 已从工作区移除，不能从回收站、旧时间戳路径或 `legacy_pilot` 恢复后
混入新结果。当前唯一保留的结果根为
`artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal`。
