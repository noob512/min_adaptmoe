#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_PACKAGE="${REPO_ROOT}/workloads/vllm/vllm/vllm"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-e7a4db7f49f7b200316a762074476c41d3c1de78}"
TARGET="${1:-/tmp/vllm_native_baseline_${EXPECTED_COMMIT:0:8}_$(date +%Y%m%d_%H%M%S)}"

if [[ -e "${TARGET}" ]]; then
  echo "error: target already exists: ${TARGET}" >&2
  exit 1
fi
if ! find "${SOURCE_PACKAGE}" -type f -name '*.so' -print -quit | grep -q .; then
  echo "error: no compiled vLLM extension found under ${SOURCE_PACKAGE}" >&2
  exit 1
fi

git -C "${REPO_ROOT}" worktree add --detach "${TARGET}" "${EXPECTED_COMMIT}"
TARGET_PACKAGE="${TARGET}/workloads/vllm/vllm/vllm"
while IFS= read -r artifact; do
  relative="${artifact#${SOURCE_PACKAGE}/}"
  install -D -m 755 "${artifact}" "${TARGET_PACKAGE}/${relative}"
done < <(find "${SOURCE_PACKAGE}" -type f -name '*.so' -print)

if [[ -n "$(git -C "${TARGET}" status --short --untracked-files=no)" ]]; then
  echo "error: prepared source has tracked modifications: ${TARGET}" >&2
  exit 1
fi

printf 'VLLM_SOURCE_ROOT=%q\n' "${TARGET}/workloads/vllm/vllm"
printf 'VLLM_SOURCE_COMMIT=%s\n' "$(git -C "${TARGET}" rev-parse HEAD)"
