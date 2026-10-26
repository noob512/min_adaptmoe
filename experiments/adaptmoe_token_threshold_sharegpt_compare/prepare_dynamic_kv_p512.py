#!/usr/bin/env python3
"""Bind existing nested DeepSeek cold plans to 2/4/6-GiB MLA KV tiers."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / 'experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/plans_adjusted_11_17_23_29'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-num-seqs', type=int, default=16)
    parser.add_argument('--prompt-tokens', type=int, default=512)
    parser.add_argument('--small-kv-ladder', action='store_true',
                        help='Use existing nested cold8/cold11 for KV 1/2 GiB')
    args = parser.parse_args()
    assert args.max_num_seqs > 0 and args.prompt_tokens > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    block_bytes = 27 * (512 + 64) * 2 * 16
    per_layer_block_bytes = block_bytes // 27
    block_alignment = 4096 // math.gcd(4096, per_layer_block_bytes)
    tiers = []
    previous = None
    ladder = ((1,8),(2,11)) if args.small_kv_ladder else ((2,11),(4,17),(6,23))
    for gib, width in ladder:
        plan_file = (PLANS.parent/'plans/cold_plan_8_per_layer.json') if gib == 1 else PLANS / f'cold_plan_{gib}g.json'
        plan = json.loads(plan_file.read_text())
        selected = {int(k): set(v) for k, v in plan['selected_by_layer'].items()}
        assert set(selected) == set(range(1, 27)) and all(len(s) == width for s in selected.values())
        assert previous is None or all(previous[k] <= selected[k] for k in selected)
        previous = selected
        physical_blocks = gib * 2**30 // block_bytes // block_alignment * block_alignment
        tiers.append({'kv_gib_label': gib, 'target_blocks': physical_blocks - 1,
                      'active_policy_kv_bytes': (physical_blocks - 1) * block_bytes,
                      'plan_class': f'cold{width}', 'cold_experts_per_layer': width,
                      'expert_bytes': plan['selected_main_bytes'], 'plan_file': str(plan_file),
                      'plan_sha256': hashlib.sha256(plan_file.read_bytes()).hexdigest()})
    classes = {'schema': 'vllm_uvm_static_alias_plan_classes_v1', 'initial_class': tiers[0]['plan_class'],
               'classes': {t['plan_class']: {'plan_file': t['plan_file'], 'capacity_gib': t['expert_bytes']/2**30} for t in tiers},
               'metadata': {'model': 'DeepSeek-V2-Lite-Chat BF16', 'nested': True,
                            'kv_bytes_per_block': block_bytes, 'tiers': tiers}}
    save(out/'plan_classes.json', classes)
    settings = {
        'VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE': str(out/'plan_classes.json'),
        'VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES': '1',
        'VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW': '1',
        'VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW': '1',
        'VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW': '1',
        'VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE': '1',
        'VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA': '1',
        'VLLM_CPU_OFFLOAD_PLAN_CLASS_INCREMENTAL_DUAL_REFRESH': '1',
        'VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE': '1',
        'VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION': '0',
        'VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW': '1',
        'VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW': '1',
        'VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE': '0',
        'VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE': 'driver_release_cpu',
        'VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE': 'post_warmup',
        'VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE': '0',
        'VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE': '0',
        'VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE': '0',
        'VLLM_UVM_DYNAMIC_EXPERT_AUTO_REBALANCE_ENABLE': '0',
        'VLLM_UVM_EXPERT_ENVELOPE_ENABLE': '1',
        'VLLM_UVM_EXPERT_ENVELOPE_BYTES': '524288',
        'VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE': 'remote_read',
        'VLLM_UVM_WEIGHT_EXECUTOR': 'cpp_batch',
        'VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE': '0',
        'VLLM_UVM_KV_CACHE_MANAGED': '1',
        'VLLM_UVM_KV_CACHE_LAZY_RESERVE': '1',
        'VLLM_UVM_KV_RUNTIME_ENABLE': '1',
        'VLLM_UVM_KV_RUNTIME_MODE': 'enforce',
        'VLLM_UVM_KV_DYNAMIC_POOL_ENABLE': '1',
        'VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE': '__CASE_DIR__/kv_dynamic_pool.control',
        'VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS': str(tiers[0]['target_blocks']),
        'VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS': str(tiers[0]['target_blocks']),
        'VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS': '16',
        'VLLM_UVM_KV_BACKUP_RESERVE_TARGET_MAP': json.dumps({str(t['target_blocks']):16 for t in tiers}),
        'VLLM_UVM_KV_BACKUP_GROW_BLOCKS': str(tiers[1]['target_blocks']-tiers[0]['target_blocks']),
        'VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE': '0',
        'VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE': '0',
        'VLLM_UVM_KV_AUTO_SCALE_ENABLE': '1',
        'VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE': '1',
        'VLLM_UVM_KV_AUTO_SCALE_SYNCHRONIZED_TIER_STEPS': '1',
        'VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY': '0',
        'VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP': json.dumps({str(t['target_blocks']):t['plan_class'] for t in tiers}),
        'VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS': ','.join(str(t['target_blocks']) for t in tiers),
        'VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW': '1',
        'VLLM_UVM_KV_AUTO_SCALE_RESUME_PREFILL_BEFORE_PLAN_CLASS': '1',
        'VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK': '3',
        'VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR': '2',
        'VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR': '2',
        'VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS': str(tiers[0]['target_blocks']),
        'VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS': str(tiers[-1]['target_blocks']),
        'VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS': '16',
        'VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK': '0.8',
        'VLLM_UVM_KV_AUTO_SCALE_ACTIVE_HIGH_WATERMARK_GROW': '1',
        'VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS': '4',
        'VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS': str(tiers[1]['target_blocks']-tiers[0]['target_blocks']),
        'VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE': '1',
        'VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE': '1',
        'VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR': 'cpp_batch',
        'VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC': '1',
        'VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR': 'cpp_batch',
        'VLLM_UVM_KV_ADAPTIVE_FAIL_FAST': '1',
        'VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST': '1',
        'VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES': '0',
        'VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE': '0',
        'VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE': '0',
        'VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE': '0',
    }
    save(out/'runtime_settings.json', settings)
    manifest = {'tiers': tiers, 'kv_block_tokens':16, 'kv_block_bytes':block_bytes,
                'virtual_pool_budget_bytes':tiers[-1]['kv_gib_label']*2**30,
                'physical_pool_blocks_including_null': tiers[-1]['target_blocks']+1,
                'per_layer_block_bytes':per_layer_block_bytes,
                'migration_page_bytes':4096, 'alignment_in_blocks':block_alignment,
                'max_num_seqs':args.max_num_seqs,
                'prompt_tokens':args.prompt_tokens, 'output_tokens':16,
                'max_active_trace_tokens_bound':min(args.max_num_seqs,128)*(args.prompt_tokens+16),
                'max_active_trace_kv_bytes_bound':min(args.max_num_seqs,128)*math.ceil((args.prompt_tokens+16)/16)*block_bytes,
                'growth_trigger':'backup/preemption/prediction or 80% actual active pressure',
                'initial_plan_matches_historical_fixed':not args.small_kv_ladder,
                'no_manual_tier_forcing':True}
    save(out/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
