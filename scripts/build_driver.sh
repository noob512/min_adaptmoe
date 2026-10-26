#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
driver_output="${DRIVER_BUILD_DIR:-${project_root}/.build/nvidia-595.71.05}"
prepare_args=(--output "${driver_output}")
if [[ -n "${NVIDIA_SOURCE_ARCHIVE:-}" ]]; then
  prepare_args+=(--source-archive "${NVIDIA_SOURCE_ARCHIVE}")
fi
python3 "${project_root}/scripts/prepare_driver.py" "${prepare_args[@]}"
make -C "${driver_output}" modules -j"${JOBS:-4}" "$@"
echo "Built UVM module: ${driver_output}/kernel-open/nvidia-uvm.ko"
