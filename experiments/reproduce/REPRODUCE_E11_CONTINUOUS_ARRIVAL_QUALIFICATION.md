# E11 continuous-arrival 当前复现状态

最后核验：2026-09-02。E11已按用户要求终止，当前不提供新的GPU运行命令；此前为诊断
E11加入的实验性修改已撤销。已有artifact保留用于审计，不能把任何失败臂拼成正式结果。

## 最新完整attempt

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e11-continuous-arrival-20260902-attempt-09-remote-park-full-dual-strict-qualification
```

离线复核：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e11-continuous-arrival-20260902-attempt-09-remote-park-full-dual-strict-qualification
jq '{status,errors,arms,correctness_comparison}' "$ROOT/verification.json"
```

该轮`phase_epoch`、`remote_phase_epoch`和`remote_continuous`均完成8/8请求；phase epoch
将mixed step降为0，remote continuous出现1个mixed step，hybrid在mixed step正确
fail-closed。但是phase-specific双Graph与两个remote-only对照的完整token IDs不一致，
顶层verifier为`status=error`。

attempt-10--16仅为单臂定因，最终attempt-16仍为`status=error`且没有补齐其他三臂；它们
不能替代attempt-09或晋升为正式结果。E11当前结论是“调度行为得到观察，但correctness门
未通过且实验已暂停”，没有可发布的性能排名。
