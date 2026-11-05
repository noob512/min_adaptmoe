# 复现 E5 当前模块stable/non-stable VA配对

最后核验：2026-09-02。两个arm使用相同Qwen3模型、layers 0--3、8个Parameter、
2,415,919,104 B逻辑alias和128 MiB物理KV，各运行10个独立冷进程。

## 运行

```bash
cd /home/ubuntu/nvidia-uvm-gpu

STABLE_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e5-destroy-rebuild-current-module-NEW-attempt-01-formal
NONSTABLE_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e5-nonstable-va-current-module-NEW-attempt-01-formal
COMPARE_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e5-stable-vs-nonstable-current-module-NEW-attempt-01

RUN_ROOT="$STABLE_ROOT" REPEATS=10 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e5_destroy_rebuild_formal.sh

RUN_ROOT="$NONSTABLE_ROOT" REPEATS=10 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e5_nonstable_va_recapture_formal.sh

mkdir -p "$COMPARE_ROOT"
workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/compare_e5_stable_vs_nonstable_va.py \
  "$STABLE_ROOT" "$NONSTABLE_ROOT" --output "$COMPARE_ROOT/comparison.json"
```

所有三个目录都必须不存在；失败attempt不得覆盖。两个arm verifier及comparison必须为
`status=ok`；stable臂8个Parameter VA保持不变，non-stable臂每代8/8 VA改变并重捕获
Graph；B1/B4输出hash必须跨臂一致。

## 当前权威结果

```text
stable     artifacts/e5-destroy-rebuild-current-module-20260902-attempt-01-formal
nonstable  artifacts/e5-nonstable-va-current-module-20260902-attempt-01-formal
comparison artifacts/e5-stable-vs-nonstable-current-module-20260902-attempt-01/comparison.json
```

| Batch | stable rebuild+recapture | non-stable reallocate+recapture | non-stable/stable |
|---:|---:|---:|---:|
| B1 | 9008.19 ms | 13712.16 ms | 1.522× |
| B4 | 8399.83 ms | 11499.47 ms | 1.369× |

两臂各10/10冷进程通过，输出一致、fault scan和资源释放通过。Graph/eager、
PARK/RESUME和sequential/grouped PTE的完整消融仍来自较早模块，不能与本轮两臂合并成
同一轮统计实验。
