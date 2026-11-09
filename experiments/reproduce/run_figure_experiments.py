#!/usr/bin/env python3
"""Run one round of the GPU experiments backing the stored paper figure tables.

Use --list for the figure-to-command map. Each job starts new cold processes;
within-process samples prescribed by the original protocol are preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from figure_run_report import inspect_run

ROOT = Path(__file__).resolve().parents[2]
EVAL = "experiments/eurosys_moe_offload_evaluation"
E13 = f"{EVAL}/e13/run_e13_kv_adaptive_matrix.sh"
INPUT = f"{EVAL}/inputs/generated/eurosys-pa-v2-20260824"


@dataclass
class Job:
    name: str
    script: str
    figures: tuple[str, ...]
    model: str = "QWEN_MODEL"
    arguments: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    positional_output: bool = False


def jobs() -> list[Job]:
    result = [
        Job("page-table-parent", f"{EVAL}/run_e12_dual_view_soak_qualification.sh",
            ("fig_pd_page_table_switch", "tab_pd_lifecycle_invariants"), model="",
            env={"CYCLES": "1000", "MINIMUM_SECONDS": "0"}),
        Job("page-table-leaf", "experiments/uvm_static_alias_original_va/run_full_scale_page_table_probe.sh",
            ("fig_pd_page_table_switch",), model=""),
        Job("placement", f"{EVAL}/e14/run_e14.sh", ("fig_e14_placement_scaling",),
            arguments=("qualification",), env={
                "E14_CONFIG": f"{EVAL}/e14/config_request_specific.json",
                "E14_COLD_PLAN": f"{EVAL}/e14/inputs/request_specific_cold29_plan.json"}),
        Job("e6-batched", f"{EVAL}/run_formal_e6_difference.sh", ("fig_batched_plan_update",),
            model="DEEPSEEK_MODEL", env={"MODE": "qualification", "PROCESS_BLOCKS": "1"}),
        Job("e6-full", f"{EVAL}/run_e6_full_rebuild_qualification.sh", ("fig_batched_plan_update",),
            model="DEEPSEEK_MODEL"),
        Job("e6-serial", f"{EVAL}/run_e6_serial_wave_qualification.sh", ("fig_batched_plan_update",),
            model="DEEPSEEK_MODEL"),
        Job("e5-graph", f"{EVAL}/run_e5_e2e_graph_once.sh", ("fig_core_mechanism_ablation",),
            arguments=("formal",)),
        Job("e5-four-paths", "experiments/vllm_uvm_cold5_alias_prefill/run_experiment.sh",
            ("fig_core_mechanism_ablation",), arguments=("qualify",),
            env={"PERFORMANCE_COPY_TRACE": "0", "WARMUP_DECODE_STEPS": "5",
                 "MEASURED_DECODE_STEPS": "8"}),
        Job("e5-prebuild-on", "experiments/vllm_uvm_dual_topology_graph/run_experiment.sh",
            ("fig_core_mechanism_ablation",), arguments=("formal",), env={
                "FORMAL_PERFORMANCE_REPEATS": "1", "KV_CACHE_BYTES": "134217728",
                "PTE_MODE": "4k", "STATIC_ALIAS_DUAL_VIEW_ENABLE": "1",
                "STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL": "1"}),
        Job("e5-prebuild-off", "experiments/vllm_uvm_dual_topology_graph/run_experiment.sh",
            ("fig_core_mechanism_ablation",), arguments=("formal",), env={
                "FORMAL_PERFORMANCE_REPEATS": "1", "KV_CACHE_BYTES": "134217728",
                "PTE_MODE": "4k", "STATIC_ALIAS_DUAL_VIEW_ENABLE": "0",
                "STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL": "0"}),
        Job("e5-stable", f"{EVAL}/run_e5_destroy_rebuild_formal.sh", ("fig_core_mechanism_ablation",),
            env={"REPEATS": "1"}),
        Job("e5-nonstable", f"{EVAL}/run_e5_nonstable_va_recapture_formal.sh", ("fig_core_mechanism_ablation",),
            env={"REPEATS": "1"}),
        Job("kv-lifecycle", E13, ("fig_plan_memory_accounting",), arguments=("qualification",)),
        Job("kv-roundtrip", E13, ("fig_fixed_plan_tradeoff", "fig_adaptive_steady_state"),
            arguments=("short-graph-roundtrip-once",)),
        Job("kv-rag", E13, ("fig_e13_rag_pressure_throughput",),
            arguments=("rag-concurrency-sweep-once",)),
        Job("kv-partition", E13, ("fig_kv_expert_partition_crossover",),
            arguments=("partition-motivation",)),
        Job("stock-uvm", f"{EVAL}/run_h_vllm_stock_uvm.sh",
            ("fig_stock_uvm_demand_paging", "fig_stock_uvm_vs_native_uva", "fig_stock_uvm_vs_native_uva_e2e"),
            model="DEEPSEEK_MODEL", positional_output=True),
        Job("qwen-stock-kv6", f"{EVAL}/run_qwen_kv6_uva_vs_uvm.sh",
            ("fig_qwen_stock_uvm_vs_native_uva",), positional_output=True),
    ]
    for prompt, batch in ((128, 8), (512, 16), (512, 32)):
        workload = {128: "short_dialogue", 512: "ordinary_chat"}[prompt]
        result.append(Job(f"pressure-p{prompt}-c{batch}",
            "experiments/adaptmoe_token_threshold_sharegpt_compare/run_compare.sh",
            ("fig_adaptmoe_pressure_throughput",), model="DEEPSEEK_MODEL", env={
                "SAMPLES": "1", "PARK_TOKEN_THRESHOLD": "16", "RESUME_TOKEN_THRESHOLD": "128",
                "FIXTURE": f"{INPUT}/deepseek_v2_lite_bf16/evaluation_{workload}.json",
                "INITIAL_REQUESTS": str(batch // 2), "LATE_REQUESTS": str(batch // 2),
                "MAX_NUM_SEQS": str(batch), "MAX_MODEL_LEN": "512" if prompt == 128 else "576",
                "MAX_NUM_BATCHED_TOKENS": str(min(8192, prompt * batch)), "OUTPUT_TOKENS": "16",
                "KV_CACHE_BYTES": str(167772160 if batch == 8 else 536870912),
                "RUNTIME_HOT_RESIDENCY_AUDIT": "0"}))
    deepseek_figs = ("fig3a_deepseek_performance", "fig_e2e_six_systems_normalized",
                     "fig_e2e_six_systems_normalized_bars")
    for batch in (1, 16):
        for arm, label, graph in (("ours", "adaptmoe", "FULL_SEPARATE"),
                ("native_pinned_uva", "uva-decode-graph", "FULL_DECODE_ONLY"),
                ("native_pinned_uva", "uva-dual-graph", "FULL_SEPARATE"),
                ("native_copy_on_forward", "copy-eager", "NONE"),
                ("native_copy_on_forward", "copy-dual-graph", "FULL_SEPARATE"),
                ("ktransformers", "kt", "NONE")):
            profile_kind = "native_copy_full_graph" if arm == "native_copy_on_forward" else "offload_plus_one_expert"
            extra = {"E2_C16_PROFILE": f"{EVAL}/configs/e2_c{batch}_near_equal_{profile_kind}.json",
                "E2_DIRECT_CONCURRENCY": str(batch), "E2_C16_SYSTEMS": arm,
                "E2_C16_CUDAGRAPH_MODE": graph, "E2_C16_NATIVE_COPY_STATIC_BANK": "0",
                "E2_KT_CPU_THREADS": "7", "E2_KT_CPU_EXPERTS_BY_WORKLOAD":
                    '{"short_dialogue":2,"ordinary_chat":' + str(2 if batch == 1 else 6) +
                    ',"rag_summary":' + str(2 if batch == 1 else 20) +
                    ',"long_generation":' + str(2 if batch == 1 else 6) + '}',
                "E2_KT_GPU_PREFILL_TOKEN_THRESHOLD_BY_WORKLOAD":
                    '{"short_dialogue":0,"ordinary_chat":0,"rag_summary":' +
                    str(0 if batch == 1 else 4096) + ',"long_generation":0}'}
            result.append(Job(f"deepseek-c{batch}-{label}", f"{EVAL}/run_e2_c16_direct_batch_matrix.sh",
                deepseek_figs, model="DEEPSEEK_MODEL", env=extra, positional_output=True))
    qwen_figs = ("fig3b_qwen_performance", "fig_phase_path_preference",
                 "fig_e2e_six_systems_normalized", "fig_e2e_six_systems_normalized_bars")
    result.append(Job("qwen-adaptmoe", f"{EVAL}/run_e3_near_equal_offload_plus_one_expert.sh",
        qwen_figs, positional_output=True))
    for graph in ("NONE", "FULL_SEPARATE"):
        result.append(Job(f"qwen-native-{graph.lower()}", f"{EVAL}/run_e3_native_copy_vs_uva_phase.sh",
            qwen_figs, env={"ATTENTION_BACKEND": "TRITON_ATTN", "BATCH_FILTER": "1,16",
                "ARM_FILTER": "copy,uva", "CUDAGRAPH_MODE": graph,
                "CONDITIONING_REPEATS": "1", "MEASUREMENT_REPEATS": "1"}, positional_output=True))
    result.append(Job("qwen-kt", f"{EVAL}/run_e3_ktransformers_c1_c16_baseline.sh", qwen_figs,
        env={"CPU_THREADS": "7", "ATTENTION_BACKEND": "flashinfer",
             "ATTENTION_BACKEND_RAG_B16": "triton", "GPU_PREFILL_TOKEN_THRESHOLD": "0",
             "GPU_PREFILL_TOKEN_THRESHOLD_MEDIUM_B16": "16384",
             "GPU_PREFILL_TOKEN_THRESHOLD_RAG_B16": "4096",
             "DISABLE_OVERLAP_SCHEDULE": "0", "E3_KT_SKIP_VERIFY": "1"}, positional_output=True))
    result.append(Job("legacy-six-systems-c1", f"{EVAL}/run_c1_mechanism_runtime_hot_audit_off_matrix.sh",
        ("fig_e2e_six_systems",), model="DEEPSEEK_MODEL",
        env={"MATRIX_SINGLE_ATTEMPT": "1"}, positional_output=True))
    result.append(Job("legacy-six-systems-c16", f"{EVAL}/run_e2_c16_direct_batch_matrix.sh",
        ("fig_e2e_six_systems",), model="DEEPSEEK_MODEL", env={
            "E2_C16_PROFILE": f"{EVAL}/configs/e2_c16_direct_batch_matrix.json",
            "E2_DIRECT_CONCURRENCY": "16",
            "E2_C16_SYSTEMS": "native_all_gpu,native_pinned_uva,native_copy_on_forward,ours,ktransformers,moe_infinity",
            "E2_C16_CUDAGRAPH_MODE": "NONE"}, positional_output=True))
    return result


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_catalogue(catalogue: list[Job]) -> None:
    names = [job.name for job in catalogue]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate experiment job names")
    for job in catalogue:
        if not (ROOT / job.script).is_file():
            raise FileNotFoundError(job.script)
        for value in (job.env or {}).values():
            if value.startswith("experiments/") and not (ROOT / value).exists():
                raise FileNotFoundError(value)
    mapped = {figure for job in catalogue for figure in job.figures}
    expected = {path.stem for path in (ROOT / "evaluation/figures/data").glob("*.csv")}
    if mapped != expected:
        raise ValueError(f"Figure mapping mismatch: missing={sorted(expected - mapped)}, "
                         f"unknown={sorted(mapped - expected)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", nargs="+", help="Job names from --list")
    args = parser.parse_args()
    catalogue = jobs()
    validate_catalogue(catalogue)
    if args.only:
        unknown = set(args.only) - {j.name for j in catalogue}
        if unknown:
            parser.error(f"Unknown jobs: {sorted(unknown)}")
        catalogue = [j for j in catalogue if j.name in args.only]
    if args.list:
        for j in catalogue:
            print(json.dumps(j.__dict__))
        return 0
    if not args.output:
        parser.error("--output must be a new output directory")
    if os.environ.get("ALLOW_GPU_RUN") != "1":
        parser.error("Set ALLOW_GPU_RUN=1 for GPU experiments")
    for variable in sorted({job.model for job in catalogue if job.model}):
        snapshot = os.environ.get(variable)
        if not snapshot or not (Path(snapshot) / "config.json").is_file():
            parser.error(f"Set {variable} to a local model snapshot containing config.json")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    # This lock covers native and custom jobs in this suite. Individual
    # historical harnesses retain their own module-cycle locks.
    lock = open("/tmp/adaptmoe-paper-figure-experiments.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "workloads/vllm/vllm") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    python = environment.get("VLLM_PY", str(ROOT / "workloads/vllm/vllm/.venv/bin/python"))
    identity = subprocess.check_output([python, "-c",
        "import json,vllm,torch; print(json.dumps({'vllm':vllm.__file__,'torch':torch.__version__}))"],
        env=environment, cwd=ROOT, text=True)
    (output / "runtime-import.txt").write_text(identity)
    if str(ROOT / "workloads/vllm/vllm/vllm") not in identity:
        raise RuntimeError("vLLM import escaped the current source checkout")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    (output / "source-diff.patch").write_bytes(patch)
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    (output / "source-status.txt").write_text(status)
    source_paths = {"experiments/reproduce/run_figure_experiments.py",
                    "experiments/reproduce/figure_run_report.py"}
    source_paths.update(job.script for job in catalogue)
    source_paths.update(subprocess.check_output(
        ["git", "ls-files", "--modified", "--others", "--exclude-standard"],
        cwd=ROOT, text=True).splitlines())
    fingerprints = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in sorted(source_paths)
                    if (ROOT / name).is_file() and not (ROOT / name).is_symlink()}
    write_json(output / "source-fingerprints.json", fingerprints)
    receipt = {"schema": "adaptmoe_paper_figure_reproduction_v1",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": git_head, "source_diff_sha256": hashlib.sha256(patch).hexdigest(),
        "source_fingerprints": "source-fingerprints.json",
        "scope": "one top-level round; protocol warmup and within-process samples preserved",
        "requested_jobs": [job.name for job in catalogue], "jobs": []}
    for job in catalogue:
        destination = output / job.name
        env = environment.copy()
        if job.env:
            env.update({k: str(ROOT / v) if v.startswith("experiments/") else v for k, v in job.env.items()})
        if job.model:
            if not env.get(job.model):
                raise RuntimeError(f"Set {job.model} to the local model snapshot")
            env["MODEL"] = env[job.model]
        env["RUN_ROOT"] = str(destination)
        command = ["bash", str(ROOT / job.script), *job.arguments]
        if job.positional_output:
            command.append(str(destination))
        entry = {"name": job.name, "figures": job.figures, "command": command,
            "environment": {k: env[k] for k in sorted(set(job.env or {}) |
                {"RUN_ROOT", "MODEL", "PYTHONPATH", "DEEPSEEK_MODEL", "QWEN_MODEL",
                 "SHAREGPT_DATASET", "FIGURE_PYTHON", "MOEI_OFFLOAD", "MOEI_RECEIPT",
                 "PYTORCH_ALLOC_CONF", "E2_C16_WORKLOADS"}) if k in env},
            "runner_sha256": hashlib.sha256((ROOT / job.script).read_bytes()).hexdigest(),
            "started_unix": time.time(), "status": "running"}
        receipt["jobs"].append(entry)
        write_json(output / "execution.json", receipt)
        print(f"START {job.name}: {shlex.join(command)}", flush=True)
        with (output / f"{job.name}.log").open("w") as log:
            rc = subprocess.call(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=rc, completed_unix=time.time(), status="completed" if rc == 0 else "failed")
        entry["verification_files"] = [str(p.relative_to(output)) for p in destination.rglob("verification.json")]
        entry["observations"] = inspect_run(destination)
        if rc == 0 and entry["observations"]["unsuccessful_records"]:
            entry["status"] = "completed_with_unsuccessful_cells"
        write_json(output / "execution.json", receipt)
        print(f"END {job.name}: returncode={rc}", flush=True)
        # A failed verifier may still leave a clean GPU. Never start another
        # job while a CUDA process or a custom module remains active.
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"], text=True).strip()
        loaded = Path("/sys/module/nvidia_uvm/srcversion").read_text().strip()
        stock = subprocess.check_output(["modinfo", "-F", "srcversion", "nvidia_uvm"], text=True).strip()
        if used != "0" or loaded != stock:
            receipt["stopped_reason"] = "GPU not empty or stock UVM not restored"
            write_json(output / "execution.json", receipt)
            break
    receipt["status"] = "ok" if len(receipt["jobs"]) == len(catalogue) and all(
        j["status"] == "completed" for j in receipt["jobs"]) else "incomplete"
    write_json(output / "execution.json", receipt)
    return 0 if receipt["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
