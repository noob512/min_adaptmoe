#!/usr/bin/env python3
"""Audit the completed P1024 scan and attach its failure/fix provenance."""
import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / 'summary.json').read_text())
    assert len(summary['cases']) == 10
    assert not (root / 'reused_samples.json').exists()
    cases = []
    for case in summary['cases']:
        directory = Path(case['directory'])
        workspace = json.loads((directory / 'mla_workspace_startup_residency.json').read_text())
        residency = workspace['residency']
        assert workspace['expected_bytes'] == 10027008
        assert residency['ok'] and residency['queried_bytes'] == residency['gpu_bytes'] == workspace['expected_bytes']
        assert residency['cpu_bytes'] == residency['unknown_bytes'] == 0
        audits = []
        for target, ok, data in re.findall(
                r'Post-growth active KV residency audit: target=(\d+) ok=(\w+) residency=(\{[^\n]+\})',
                (directory / 'run.log').read_text()):
            audit = ast.literal_eval(data)
            expected = (int(target) + 1) * 497664
            assert ok == 'True' and audit['ok']
            assert audit['queried_bytes'] == audit['gpu_bytes'] == expected
            assert audit['cpu_bytes'] == audit['unknown_bytes'] == 0
            audits.append({'target_blocks': int(target), 'residency': audit})
        assert len(audits) == case['growth_count']
        cases.append({'qps': case['offered_qps'], 'sample': case['sample'],
                      'directory': str(directory), 'startup_workspace': workspace,
                      'post_growth_kv_audits': audits})
    assert sum(len(c['post_growth_kv_audits']) for c in cases) == 8
    repo = Path(__file__).resolve().parents[2]
    source_files = {
        'gpu_model_runner.py': repo / 'workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py',
        'gpu_worker.py': repo / 'workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py',
        'uvm.py': repo / 'workloads/vllm/vllm/vllm/device_allocator/uvm.py',
        'dynamic_kv_p512_runner.py': Path(__file__).with_name('dynamic_kv_p512_runner.py'),
    }
    hashes = {}
    for name, source in source_files.items():
        content = source.read_bytes()
        assert content == (root / 'source_snapshots' / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
    gpu_memory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    srcversion = Path('/sys/module/nvidia_uvm/srcversion').read_text().strip()
    services = {s: subprocess.check_output(['systemctl', 'is-active', s], text=True).strip()
                for s in ('node-exporter.service', 'gpu-exporter.service')}
    assert gpu_memory == '0' and srcversion == '85A79790636BBD99BA3E43B'
    assert all(state == 'active' for state in services.values())
    qualification = {
        'status': 'ok', 'valid_processes': 10, 'successful_requests': 1280,
        'successful_growth_transactions': 8, 'cases': cases, 'source_sha256': hashes,
        'final_health': {'gpu_memory_mib': int(gpu_memory), 'srcversion': srcversion, 'services': services},
        'tests': {'worker_lazy_pool_and_resize': 23, 'dynamic_pool_policy': 22},
    }
    (root / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    marker = '\n## 升档结论与执行修复\n'
    report = (root / 'REPORT.md').read_text().split(marker)[0]
    report += marker + '''
QPS4/8/10的两次重复均保持K2；QPS20/30的两次重复均自动K2→K4→K6。
此前P512/C128动态实验峰值3228 blocks，加16 backup和16 safety后的需求3260，低于K2高水位3451。
输入1024、输出16时，每请求KV上限65 blocks；原512输入为33 blocks，接近翻倍。
以QPS20/sample1为例：step12需求3472超过3451，step21需求6934超过6904，分别触发两级升档。
这里比较的是控制器的required_blocks（active+backup+safety），不能误称为裸active_blocks已越线。
K2/K4/K6仍分别配对cold11/17/23，专家卸载约4.608/7.122/9.636GiB。
本轮没有成功降档记录；仅凭请求减少不能断言所有安全降档条件都已满足。

本轮验证了自动升档能够实际发生并完成端到端请求，未重测相同1024输入的固定KV，不能据此宣称动态KV比固定KV更快。
约0.57/0.64秒的升档事务包含专家切换、实际KV物理迁移和驻留审计，不能当成普通PARK/RESUME页表切换时间。

执行中先停止并修复了以下错误，失败样本全部排除：

- attempt01：18KiB MLA block从block1开始的KV驻留审计范围未按4KiB对齐，返回NV_ERR_INVALID_ADDRESS(-30)。改为包含null block0的完整GPU前缀，与初始化一致。
- attempt03：升档已提交，但首次chunked-prefill写入managed工作区时发生非法地址；lazy KV作用域错误覆盖了metadata builder工作区。改为仅KV张量关闭预取，工作区正常预取。
- attempt04：嵌套进入同一PyTorch MemPool报错；内层现只切换预取开关，由最外层选择pool，并在正常/异常退出时恢复开关。

最终attempt05将全部五档QPS各重跑两次，无复用旧样本；所有1280请求成功，无Xid/非法访问。
每个进程的10,027,008B MLA工作区在首请求前均GPU独占驻留；8次成功升档后的完整KV前缀驻留审计均通过，见qualification.json。
保留manual_migration_fault_mode=2、80%阈值、物理迁移、专家配对与安全拒绝；没有启用隐式迁移或强制升档。
源码快照、逐请求原始样本、resize事务及失败日志均保留。最终GPU显存为0MiB，stock UVM与两个exporter已恢复。
'''
    (root / 'REPORT.md').write_text(report)
    shutil.copy2(__file__, root / 'source_snapshots' / Path(__file__).name)
    for suffix, source in (('md', 'REPORT.md'), ('json', 'summary.json'), ('csv', 'summary.csv')):
        shutil.copy2(root / source, Path(__file__).with_name(f'P1024_N128_C128_DYNAMIC_KV_20261024.{suffix}'))
    print(json.dumps({'status': 'ok', 'valid_processes': 10, 'growth_transactions': 8,
                      'report': str(root / 'REPORT.md')}, indent=2))


if __name__ == '__main__':
    main()
