# E8 MoE-Infinity cache/prefetch复现状态

最后核验：2026-09-02。活动MoE-Infinity HEAD为
`9741f80b53374f0d98e7bd8f4a777b816bf68de8`，父仓库gitlink仍指向
`92c2bc9266e33bbed22405356ca7ca7e71ae8974`；在更新gitlink和生成fresh receipt前，
fresh clone也不能复现当前活动代码。

当前只有provenance preflight可用：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/broad_offload_compare/run_moe_infinity_qualification.sh preflight
```

截至2026-09-02，该命令仍因缺少与活动MoE-Infinity commit、扩展ELF和约30 GiB store一致的
fresh build receipt而返回1。不得绕过commit、patch、ELF或receipt hash门运行性能实验。

因此E8目前没有成功性能实验可复现。本文件记录的是合法阻塞入口，不代表E8算法性能失败。
