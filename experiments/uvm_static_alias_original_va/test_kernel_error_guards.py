# SPDX-License-Identifier: MIT

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VA_BLOCK_SOURCE = (
    REPO_ROOT
    / "kernel-module"
    / "nvidia-module"
    / "kernel-open"
    / "nvidia-uvm"
    / "uvm_va_block.c"
)


def test_partial_2m_unmap_validates_backing_before_gpu_submission() -> None:
    source = VA_BLOCK_SOURCE.read_text(encoding="utf-8")
    helper_start = source.index("static NV_STATUS block_validate_2m_split_backing(")
    allocation_start = source.index("static NV_STATUS block_alloc_ptes_new_state(")
    assert allocation_start < helper_start

    helper_end = source.index(
        "// Make sure that GMMU PDEs down to PDE1 are populated", helper_start
    )
    helper = source[helper_start:helper_end]
    assert "reason=missing_logical_residency" in helper
    assert "reason=mixed_residency" in helper
    assert "reason=missing_physical_backing" in helper
    assert "uvm_page_mask_subset(&remaining_pages" in helper
    assert "&block->cpu.allocated" in helper
    assert "num_gpu_chunks = block_num_gpu_chunks" in helper
    assert "resident_gpu_state->chunks[chunk_index]" in helper
    assert "for_each_va_block_page_in_region" not in helper
    assert helper.count("return NV_ERR_INVALID_STATE;") >= 4

    unmap_start = source.index("static NV_STATUS block_unmap_gpu(")
    unmap_end = source.index("NV_STATUS uvm_va_block_unmap(", unmap_start)
    unmap = source[unmap_start:unmap_end]
    guard = unmap.index("block_validate_2m_split_backing(")
    allocate = unmap.index("block_alloc_ptes_new_state(")
    submit = unmap.index("uvm_push_begin_acquire(")
    split = unmap.index("block_gpu_unmap_split_2m(")

    assert guard < allocate < submit < split
    assert "if (status != NV_OK)\n            return status;" in unmap[guard:allocate]


def test_immutable_shadow_partial_release_presplits_before_free() -> None:
    source = VA_BLOCK_SOURCE.read_text(encoding="utf-8")

    helper_start = source.index(
        "static NV_STATUS block_presplit_2m_for_partial_shadow_cutover("
    )
    cutover_start = source.index(
        "static NV_STATUS block_cutover_to_existing_cpu_shadow("
    )
    release_start = source.index(
        "NV_STATUS uvm_va_block_release_gpu_subchunks_to_cpu("
    )
    helper = source[helper_start:cutover_start]
    cutover = source[cutover_start:release_start]

    assert (
        "uvm_va_block_region_size(region) == uvm_va_block_size(va_block)"
        in helper
    )
    validate = helper.index("block_validate_2m_split_backing(")
    allocate = helper.index("block_alloc_ptes_new_state(")
    submit = helper.index("uvm_push_begin_acquire(")
    split = helper.index("block_gpu_split_2m_to_resident(", submit)
    wait = helper.index("uvm_push_end_and_wait(", split)
    verify = helper.index("gpu_state->pte_is_2m", wait)
    assert validate < allocate < submit < split < wait < verify
    # The resident-aware helper preserves complete CPU shadow backing when
    # local GPU residency is sparse; it replaces the older generic split.
    assert "resident_override = UVM_ID_CPU;" in helper[:validate]
    assert "uvm_page_mask_subset(&mapped_pages, &va_block->cpu.resident)" in helper
    assert "uvm_page_mask_subset(&mapped_pages, &va_block->cpu.allocated)" in helper
    assert "resident_override," in helper[split:wait]

    presplit = cutover.index("block_presplit_2m_for_partial_shadow_cutover(")
    make_resident = cutover.index("uvm_va_block_make_resident(")
    post_resplit = cutover.index(
        "block_presplit_2m_for_partial_shadow_cutover(", make_resident
    )
    postcheck = cutover.index("gpu_state->pte_is_2m", post_resplit)
    assert presplit < make_resident < post_resplit < postcheck
    # The failure branch also emits telemetry before leaving the cutover.
    failure_guard = cutover.index("if (status != NV_OK)", presplit)
    failure_exit = cutover.index("goto out;", failure_guard)
    assert presplit < failure_guard < failure_exit < make_resident

    release_end = source.index("return status;", release_start)
    release = source[release_start:release_end]
    cutover_call = release.index("block_cutover_to_existing_cpu_shadow(")
    chunk_conversion = release.index("block_convert_gpu_chunks_to_fixed_size(")
    chunk_free = release.index("uvm_pmm_gpu_free(")
    assert cutover_call < chunk_conversion < chunk_free


def test_immutable_shadow_presplit_has_runtime_telemetry() -> None:
    source = VA_BLOCK_SOURCE.read_text(encoding="utf-8")
    for name in (
        "immutable_shadow_presplit_calls",
        "immutable_shadow_presplit_successes",
        "immutable_shadow_presplit_already_lower",
        "immutable_shadow_presplit_failures",
        "immutable_shadow_presplit_ns",
    ):
        assert f'g_uvm_{name}' in source
        assert f'"{name}' in source
        assert f"atomic64_set(&g_uvm_{name}, 0);" in source


def test_immutable_shadow_partial_remote_map_cannot_remerge_2m() -> None:
    source = VA_BLOCK_SOURCE.read_text(encoding="utf-8")

    map_start = source.index("static NV_STATUS block_map_gpu_to(")
    map_end = source.index(
        "static NvU64 block_moe_slot_pte_mode_page_size(", map_start
    )
    gpu_map = source[map_start:map_end]
    compute = gpu_map.index("block_gpu_compute_new_pte_state(")
    force_lower = gpu_map.index(
        "if (forbid_2m_merge && new_pte_state->pte_is_2m)", compute
    )
    allocate = gpu_map.index("block_alloc_ptes_new_state(", force_lower)
    assert compute < force_lower < allocate
    assert "new_pte_state->pte_is_2m = false;" in gpu_map[force_lower:allocate]
    assert "new_pte_state->needs_4k = true;" in gpu_map[force_lower:allocate]

    premap_start = source.index(
        "NV_STATUS uvm_va_block_premap_readmostly_remote_read("
    )
    premap_end = source.index(
        "NV_STATUS uvm_va_block_enqueue_readmostly_bg_promote(", premap_start
    )
    premap = source[premap_start:premap_end]
    remote_map = premap.index("block_force_remote_map_cpu_read_locked(")
    postcheck = premap.index("immutable_cpu_shadow_postmap_reject", remote_map)
    telemetry = premap.index(
        "g_uvm_immutable_shadow_postmap_checks", remote_map
    )
    assert remote_map < telemetry < postcheck
    assert "gpu_state->pte_is_2m" in premap[telemetry:postcheck]
    assert "gpu_state->page_table_range_4k.table" in premap[telemetry:postcheck]

    for name in (
        "immutable_shadow_forced_lower_maps",
        "immutable_shadow_postmap_checks",
        "immutable_shadow_postmap_failures",
    ):
        assert f"g_uvm_{name}" in source
        assert f'"{name}' in source
        assert f"atomic64_set(&g_uvm_{name}, 0);" in source


def test_dual_refresh_accepts_disjoint_hot_big_ptes_and_republishes_child() -> None:
    source = VA_BLOCK_SOURCE.read_text(encoding="utf-8")
    validate_start = source.index(
        "static NV_STATUS block_slot_alias_dual_validate_locked("
    )
    validate_end = source.index(
        "static void block_slot_alias_dual_copy_4k_table(", validate_start
    )
    validate = source[validate_start:validate_end]
    assert "if (require_ready &&" in validate
    assert "uvm_page_mask_init_from_big_ptes(" in validate
    assert "uvm_page_mask_and(&scratch, &scratch, dual_pages)" in validate

    prepare_start = source.index(
        "static NV_STATUS block_slot_alias_dual_prepare_locked("
    )
    prepare_end = source.index(
        "NV_STATUS uvm_va_block_slot_alias_dual_prepare_batch(", prepare_start
    )
    prepare = source[prepare_start:prepare_end]
    refresh = prepare.index("if (was_ready && !was_original_active)")
    republish = prepare.index("uvm_page_tree_write_pde_with_child(", refresh)
    invalidate = prepare.index("uvm_tlb_batch_invalidate(", republish)
    normalize = prepare.index("bitmap_zero(gpu_state->big_ptes", invalidate)
    assert refresh < republish < invalidate < normalize
