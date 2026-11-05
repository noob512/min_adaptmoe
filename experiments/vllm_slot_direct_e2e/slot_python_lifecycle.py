#!/usr/bin/env python3
"""Validate the real vLLM mode-8 -> mode-10 -> mode-19 Python lifecycle."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


MIB = 1024 * 1024
VA_BLOCK_BYTES = 2 * MIB
REGION_BYTES = 512 * 1024
ELEMENT_BYTES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=int, default=8)
    parser.add_argument("--release-pattern", choices=("all", "first-then-rest"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.regions <= 128:
        parser.error("--regions must be between 2 and 128")
    return args


def require_ok(label: str, result: dict[str, object], count: int) -> None:
    if (
        not bool(result.get("ok"))
        or int(result.get("success_count", -1)) != count
        or int(result.get("failure_count", -1)) != 0
        or any(status != 0 for status in result.get("entry_statuses", []))
    ):
        raise RuntimeError(f"{label} failed: {result}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "workloads" / "vllm" / "vllm"))

    import torch
    from vllm.device_allocator import uvm as uvm_module

    batch_prefetch_ranges_to_cpu = uvm_module.batch_prefetch_ranges_to_cpu
    close_uvm_slot_direct_session = uvm_module.close_uvm_slot_direct_session
    driver_slot_preallocate_ranges = uvm_module.driver_slot_preallocate_ranges
    driver_slot_release_ranges = uvm_module.driver_slot_release_ranges
    driver_slot_restore_ranges = uvm_module.driver_slot_restore_ranges
    release_uvm_local_memory_pool = uvm_module.release_uvm_local_memory_pool
    uvm_local_memory_pool = uvm_module.uvm_local_memory_pool

    torch.cuda.set_device(0)
    raw = None
    views: list[torch.Tensor] = []
    output: dict[str, object] = {
        "schema": "vllm_slot_python_lifecycle_v1",
        "status": "running",
        "regions": args.regions,
        "region_bytes": REGION_BYTES,
        "release_pattern": args.release_pattern,
    }
    try:
        raw_bytes = VA_BLOCK_BYTES + REGION_BYTES + args.regions * REGION_BYTES
        with uvm_local_memory_pool(enable_prefetch=False):
            raw = torch.empty(
                (raw_bytes + ELEMENT_BYTES - 1) // ELEMENT_BYTES,
                dtype=torch.float16,
                device="cuda",
            )

        raw_ptr = int(raw.data_ptr())
        aligned_block = (raw_ptr + VA_BLOCK_BYTES - 1) // VA_BLOCK_BYTES * VA_BLOCK_BYTES
        first_ptr = aligned_block + REGION_BYTES
        first_element = (first_ptr - raw_ptr) // ELEMENT_BYTES
        region_elements = REGION_BYTES // ELEMENT_BYTES
        for index in range(args.regions):
            view = raw.narrow(0, first_element + index * region_elements, region_elements)
            view.fill_((index + 1) / 1024.0)
            views.append(view)
        torch.cuda.synchronize()

        pointers = [int(view.data_ptr()) for view in views]
        original_pointers = list(pointers)
        ranges = [
            (pointer, REGION_BYTES, 77, 19, 1000 + index, 0)
            for index, pointer in enumerate(pointers)
        ]
        identities = [(77, 19, 1000 + index) for index in range(args.regions)]
        ordinary_ranges = [(pointer, REGION_BYTES) for pointer in pointers]
        checksums_before = [float(view.float().sum().item()) for view in views]
        torch.cuda.synchronize()

        preallocate = driver_slot_preallocate_ranges(ranges, device=0)
        require_ok("preallocate", preallocate, args.regions)

        accepted = batch_prefetch_ranges_to_cpu(ordinary_ranges)
        torch.cuda.synchronize()
        if accepted != args.regions:
            raise RuntimeError(f"CPU prefetch accepted {accepted}/{args.regions}")

        restore = driver_slot_restore_ranges(ranges, device=0)
        require_ok("restore", restore, args.regions)
        if [int(view.data_ptr()) for view in views] != original_pointers:
            raise RuntimeError("managed tensor VA changed after restore")

        checksums_slot = [float(view.float().sum().item()) for view in views]
        torch.cuda.synchronize()
        if checksums_slot != checksums_before:
            raise RuntimeError("slot-backed checksum mismatch")

        releases: list[dict[str, object]] = []
        if args.release_pattern == "all":
            release = driver_slot_release_ranges(identities, device=0)
            require_ok("release-all", release, args.regions)
            releases.append(release)
            reuse_ranges = ranges
            reuse_ordinary_ranges = ordinary_ranges
        else:
            first = driver_slot_release_ranges(identities[:1], device=0)
            require_ok("release-first", first, 1)
            releases.append(first)
            after_first = [float(view.float().sum().item()) for view in views]
            torch.cuda.synchronize()
            if after_first != checksums_before:
                raise RuntimeError("checksum mismatch after partial release")

            # Keep the remaining same-VA-block slots live while the first
            # range is made CPU-resident and reused. This catches accidental
            # large-PTE merging across an active neighbouring slot mapping.
            reuse_ranges = ranges[:1]
            reuse_ordinary_ranges = ordinary_ranges[:1]

        accepted = batch_prefetch_ranges_to_cpu(reuse_ordinary_ranges)
        torch.cuda.synchronize()
        if accepted != len(reuse_ranges):
            raise RuntimeError(
                f"reuse CPU prefetch accepted {accepted}/{len(reuse_ranges)}"
            )

        reuse_preallocate = driver_slot_preallocate_ranges(reuse_ranges, device=0)
        require_ok("reuse-preallocate", reuse_preallocate, len(reuse_ranges))
        reuse_restore = driver_slot_restore_ranges(reuse_ranges, device=0)
        require_ok("reuse-restore", reuse_restore, len(reuse_ranges))
        if [int(view.data_ptr()) for view in views] != original_pointers:
            raise RuntimeError("managed tensor VA changed after same-process reuse")
        checksums_reused = [float(view.float().sum().item()) for view in views]
        torch.cuda.synchronize()
        if checksums_reused != checksums_before:
            raise RuntimeError("slot-backed checksum mismatch after same-process reuse")

        final_release = driver_slot_release_ranges(identities, device=0)
        require_ok("final-release-all", final_release, args.regions)
        releases.append(final_release)

        checksums_released = [float(view.float().sum().item()) for view in views]
        torch.cuda.synchronize()
        if checksums_released != checksums_before:
            raise RuntimeError("checksum mismatch after mode-19 release")
        if [int(view.data_ptr()) for view in views] != original_pointers:
            raise RuntimeError("managed tensor VA changed after release")
        if uvm_module._uvm_slot_direct_owners.get(0, {}):
            raise RuntimeError(
                "mode-19 left stale low-level userspace slot ownership"
            )

        close = close_uvm_slot_direct_session(device=0)
        if not bool(close.get("ok")) or bool(close.get("closed")) is not True:
            raise RuntimeError(f"session close failed: {close}")

        output.update(
            {
                "status": "ok",
                "pointers": pointers,
                "regions_per_va_block": {
                    str(block): sum(pointer // VA_BLOCK_BYTES == block for pointer in pointers)
                    for block in sorted({pointer // VA_BLOCK_BYTES for pointer in pointers})
                },
                "preallocate": preallocate,
                "restore": restore,
                "reuse_preallocate": reuse_preallocate,
                "reuse_restore": reuse_restore,
                "releases": releases,
                "close": close,
                "checksums_equal": True,
                "va_stable": True,
                "same_process_reuse": True,
                "low_level_owner_state_clean": True,
            }
        )
    except Exception as exc:
        output["status"] = "exception"
        output["error"] = repr(exc)
        raise
    finally:
        text = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
        views.clear()
        raw = None
        gc.collect()
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            release_uvm_local_memory_pool()
        except Exception:
            if output.get("status") == "ok":
                raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
