# 约 2 GiB 等卸载量、0.5 req/s、UVA/AdaptMoE 复跑

用户将当前任务收敛为两个机制、统一约 2 GiB 卸载量、QPS=0.5；此前四机制 K4 矩阵已按要求停止，不能作为本轮完整结果。

本轮为 3 种 trace × 2 个机制，每个配置一次独立启动和一次完整回放，共 6 个配置、384 条正式请求。模型仍为 Qwen3-30B-A3B-FP8，RTX 5090，TRITON_ATTN，eager，关闭 CUDA Graph。vLLM 使用原生混合调度，C64、8192-token step budget、client connections=96，禁用 prefix cache。

Short/Medium/Long 沿用相同 64 条 ShareGPT 来源请求和顺序，prompt 前缀为 128/512/1024 tokens，均生成 512 tokens、ignore_eos。到达时刻改为 0,2,...,126 s，即恒定提供速率 0.5 req/s；不再使用上一轮 0.25→4→0.25 的压力波。4 条并发预热请求各生成 8 tokens，不计入正式窗口。

“2 GiB 档”使用项目原有 cold10 计划，每层 10 个冷专家、48 层，共 480 个。由于现有静态专家计划要求各层等宽，其实际主权重为 **2,264,924,160 B = 2.109375 GiB**，不能把它标为严格 2.000 GiB。Native UVA 使用项目已有 exact-name 选择器冻结 12 个完整 Parameter：前三层的 w13/w2、第四层 w13、前四层 qkv weight、第一层 o_proj weight，合计同为 **2,264,924,160 B**。两臂实际卸载字节差为 0。此处匹配字节数，不匹配专家坐标；UVA 包含 attention 权重。

卸载量从此前约 6 GiB 下调后，两臂都固定为 **1.5 GiB KV**（1,610,612,736 B；1023 个有效 blocks、16,368 tokens），避免沿用 K4 挤压显存。AdaptMoE 复用安全的 full-dual-view 初始化框架，只保留一个 cold10 plan class，KV 的 initial/min/max/allowed target 全部为 1023，禁用主动增长；正式期间不允许任何 KV 或专家改档。它仍按 scheduler step 总 token 数选择拓扑：≤16 remote，≥128 copy，中间保持状态。每轮诊断审计和 CUDA read gate 关闭，必要初始化安全检查保留。

QPS=64/正式回放至所有请求完成的总时长，包含尾部排空，所以实际 QPS 可以低于提供速率 0.5；TTFT、TPOT、E2E 为请求均值。TPOT 以首末输出 token 时间差除以 511 计算。本轮每配置只有一个样本，不宣称统计显著性或完整输出数值等价。

入口：

```bash
bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_mixed_2g_qps05_uva_adaptmoe.sh \
 /home/ubuntu/nvidia-uvm-gpu/experiments/eurosys_moe_offload_evaluation/sharegpt_trace/artifacts/mixed-2g-qps05-uva-adaptmoe-20261023-attempt-01
```

最终分析入口为 `analyze_mixed_2g.py`。审核六项全部成功、相同 trace hash、到达时刻、完整输出长度、实际卸载字节数、Graph 关闭、固定 KV/专家和真实 mixed step。失败立即停止并保留产物。运行期间不编辑执行中的 shell 文件。
