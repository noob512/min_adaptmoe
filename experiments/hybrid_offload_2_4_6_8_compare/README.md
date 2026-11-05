# C2/C4/C6/C8 固定 Graph Plan Class

本目录保留固定 Graph class 的实验代码和复现方法。实验验证目标是：四档 GraphExec
在启动时各捕获一次，此后改变专家容量
与物理 backing 时不再 reset/recapture；切换只处理 added/removed 专家、保留共同
backing/session、只 refill 变化 slot，且所有 Graph 可见 VA 保持不变。

## 配置

- class manifest：`inputs/graph_plan_classes.json`
- 初始 class：C8
- 默认切换序列：`C2 -> C4 -> C6 -> C8 -> C2`
- Graph 模式：`FULL_SEPARATE`
- 生命周期：`sealed <-> remote_parked`
- RTX 5090 qualification attention 路径：关闭 MLA absorption，固定
  `TRITON_ATTN`；本机可用的 `TRITON_MLA` 不支持 `FULL_SEPARATE`
- 关键环境变量：

```bash
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE=\
  experiments/hybrid_offload_2_4_6_8_compare/inputs/graph_plan_classes.json
VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1
VLLM_MLA_DISABLE=1
VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

热度异步 pinned D2H 不属于本次 qualification；这里使用冻结的四份 expert plan，
只验证稳定 envelope、跨容量 backing 和 GraphExec 复用。

## 运行

脚本会检查 GPU 空闲、加载本仓库编译的定制 UVM 模块、运行一个 vLLM 进程，并在
结束后恢复 stock 模块：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ALLOW_GPU_RUN=1 \
  experiments/hybrid_offload_2_4_6_8_compare/run_graph_plan_class_sequence.sh
```

可通过 `RUN_ROOT`、`MODEL`、`FIXTURE`、`KV_CACHE_BYTES` 和
`PLAN_CLASS_SEQUENCE` 覆盖默认值。输出包括 `result.json`、`manifest.jsonl`、模块
快照、GPU 前后状态、kernel journal 和完整日志。

## 通过条件

`result.json` 必须同时满足：

- C2/C4/C6/C8 四档均有完整 Graph certificate 和 GraphExec 表；
- 每次激活后的逻辑 selected bytes 等于对应计划字节数；
- 物理 backing bytes 等于切换事务实际物化的字节数、同一 class 重复激活时保持
  一致，并且不超过稳定 envelope；
- Parameter、upload 和 pinned source VA 全序列不变；
- Graph capture counter 在模型加载后不再增加；
- 每档请求的 greedy token IDs/hash 完全一致；
- alias session 始终存在且不为 poisoned；
- shutdown 释放全部 Graph、alias、backing 和 Python tensor 引用。

## Results

历史 qualification artifacts 和独立结果表已按清理策略移除。本目录不再保留旧实验
数值；任何新结果必须使用新的 run ID 重新生成。
