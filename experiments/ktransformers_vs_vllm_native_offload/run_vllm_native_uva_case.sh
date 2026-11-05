#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The pinned upstream vLLM revision implements cpu_offload_gb with pinned host
# allocations exposed to CUDA through UVA.  Keep the backend explicit so a
# worktree default cannot silently change the baseline mechanism.
OFFLOAD_BACKEND=uva exec bash \
  "${SCRIPT_DIR}/run_vllm_native_copy_case.sh" "$@"
