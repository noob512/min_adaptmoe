# 固定 KV、混合调度四机制复跑（2026-10-23）

按用户最新要求，本轮仅测固定 KV 的 AdaptMoE、Native UVA、Native Copy、KTransformers，三种 trace 各独立启动一次，共 12 个配置。新运行和旧的六臂实验分开保存，不复用旧性能样本。

模型为 Qwen3-30B-A3B-FP8，GPU 为 RTX 5090。全部关闭 CUDA Graph；vLLM 使用 TRITON_ATTN、eager、chunked mixed scheduling，关闭 phase cohorts。AdaptMoE 按当前 scheduler step 总 token 数选择访问拓扑：≤16 PARK/remote，≥128 RESUME/copy，中间保持原状态。固定 K4：2730 个 policy blocks、43680 tokens、约 4 GiB KV；cold28 为每层卸载 28 个专家，主权重共 5.90625 GiB。

AdaptMoE 复用已验证的安全初始化路径，在预热阶段完成 K2/cold14→K4/cold28 的事务，此后保持 K4。脚本逐 cell 审核仅有一次 K4 提交且其时间早于正式回放开始；若不满足立即停止。此初始化不计入正式 TTFT/TPOT/QPS，正式窗口不得发生容量或专家档位变化。

三种 trace 从冻结的 ShareGPT low 输入中选择相同 64 条来源请求，分别取 128/512/1024-token 前缀，统一生成 512 tokens、ignore_eos。到达过程完全相同：8 条请求以 0.25 req/s 到达，48 条以 4 req/s 到达，最后 8 条恢复至 0.25 req/s；最后到达在 72 s。这里 Long 指本组最长的 1024-token prompt，并非 8K 长上下文。到达率恢复后可能仍有积压，不预设延迟不变或已恢复低 KV 压力。

并发 C64、每 step 8192 tokens、client connections 96；每配置先执行 4 条并发预热，各生成 8 tokens。QPS 统计 64 条正式请求全部排空；TTFT、TPOT、E2E 为请求均值。每 cell 只有一次有效回放，无方差或显著性结论。

Native 两臂使用同一 5.907 GiB offload 预算，按完整 Parameter 选择；实际卸载量需要从 manifest 报告，不能声称与 AdaptMoE 坐标一致。KTransformers 使用同一 cold28 专家计划，每层 100 个 GPU 专家，KV 上限 43696 tokens，mixed chunk 开启、radix cache 关闭、full-GPU Prefill fallback 关闭。它采用不同的 scheduler、page size 和 CPU 专家算子，是系统配置对比。

复现：

```bash
bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_bounded_mixed_fixed_matrix.sh \
  /home/ubuntu/nvidia-uvm-gpu/experiments/eurosys_moe_offload_evaluation/sharegpt_trace/artifacts/bounded-mixed-fixed-four-arm-20261023-attempt-01
```

运行前完成脚本修改和语法检查；运行期间不编辑执行中的 shell 文件。发生运行错误立即停止矩阵并保留失败产物。最终核验 12 个配置、768 条成功请求、trace hash、输出 token 数、关闭 Graph、真实 mixed step、固定容量和 token 访问切换。
