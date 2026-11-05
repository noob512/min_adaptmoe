#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MOEI_ROOT="${REPO_ROOT}/MoE-Infinity"
PYTHON="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
ELF_FINGERPRINTER="${SCRIPT_DIR}/elf_runtime_fingerprint.py"
VENV_BIN="$(dirname "${PYTHON}")"
CUDA_HOME="/usr/local/cuda-12.8"
NVTX_INCLUDE_DIR="/usr/local/cuda-12.8/nsight-systems-2024.6.2/target-linux-x64/nvtx/include"
OUTPUT_DIR="${1:-}"
EXPECTED_MOEI_COMMIT="${EXPECTED_MOEI_COMMIT:-9741f80b53374f0d98e7bd8f4a777b816bf68de8}"

[[ -n "${OUTPUT_DIR}" ]] || {
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 1
}
ACTUAL_MOEI_COMMIT="$(git -C "${MOEI_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_MOEI_COMMIT}" == "${EXPECTED_MOEI_COMMIT}" ]] || {
  echo "MoE-Infinity commit drift" >&2
  exit 1
}
[[ -x "${VENV_BIN}/ninja" ]] || {
  echo "the pinned venv ninja executable is absent: ${VENV_BIN}/ninja" >&2
  exit 1
}
[[ -f "${ELF_FINGERPRINTER}" ]] || {
  echo "ELF runtime fingerprinter is absent: ${ELF_FINGERPRINTER}" >&2
  exit 1
}
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || {
  echo "the pinned CUDA compiler is absent: ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
}
[[ -f "${NVTX_INCLUDE_DIR}/nvtx3/nvtx3.hpp" ]] || {
  echo "the pinned NVTX header is absent: ${NVTX_INCLUDE_DIR}/nvtx3/nvtx3.hpp" >&2
  exit 1
}

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
BUILD_LOG="${OUTPUT_DIR}/build.log"
COMMAND_FILE="${OUTPUT_DIR}/command.txt"
STARTED_AT="$(date --iso-8601=seconds)"
PATCH_SHA256="$(git -C "${MOEI_ROOT}" diff --binary | sha256sum | awk '{print $1}')"

printf '%q ' env "PATH=${VENV_BIN}:${PATH}" MOE_BUILD_EXTENSIONS=_store \
  MOE_ENABLE_SM90=0 MOE_ENABLE_SM120=1 MAX_JOBS=14 NVTX_DISABLE=0 \
  "CUDA_HOME=${CUDA_HOME}" "NVTX_INCLUDE_DIR=${NVTX_INCLUDE_DIR}" \
  "${PYTHON}" setup.py build_ext --inplace --force >"${COMMAND_FILE}"
printf '\n' >>"${COMMAND_FILE}"

set +e
(
  cd "${MOEI_ROOT}"
  /usr/bin/time -v env \
    PATH="${VENV_BIN}:${PATH}" \
    CUDA_HOME="${CUDA_HOME}" \
    MOE_BUILD_EXTENSIONS=_store \
    MOE_ENABLE_SM90=0 \
    MOE_ENABLE_SM120=1 \
    MAX_JOBS=14 \
    NVTX_DISABLE=0 \
    NVTX_INCLUDE_DIR="${NVTX_INCLUDE_DIR}" \
    "${PYTHON}" setup.py build_ext --inplace --force
) >"${BUILD_LOG}" 2>&1
RC=$?
set -e
printf '%s\n' "${RC}" >"${OUTPUT_DIR}/returncode.txt"

EXTENSION="$(find "${MOEI_ROOT}/moe_infinity" -maxdepth 1 -type f \
  -name '_store*.so' -print -quit)"
[[ -n "${EXTENSION}" && -f "${EXTENSION}" ]] || {
  echo "rebuilt _store extension is absent" >&2
  exit 1
}
EXTENSION_SHA256="$(sha256sum "${EXTENSION}" | awk '{print $1}')"
RUNTIME_FINGERPRINT_JSON="${OUTPUT_DIR}/runtime_elf_fingerprint.json"
"${PYTHON}" "${ELF_FINGERPRINTER}" --json "${EXTENSION}" \
  > "${RUNTIME_FINGERPRINT_JSON}"
RUNTIME_ELF_SHA256="$("${PYTHON}" - "${RUNTIME_FINGERPRINT_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["runtime_elf_sha256"])
PY
)"
RUNTIME_FINGERPRINT_JSON_SHA256="$(sha256sum "${RUNTIME_FINGERPRINT_JSON}" | awk '{print $1}')"
BUILD_LOG_SHA256="$(sha256sum "${BUILD_LOG}" | awk '{print $1}')"
FINISHED_AT="$(date --iso-8601=seconds)"

"${PYTHON}" - "${OUTPUT_DIR}/store_build_receipt.json" \
  "${RC}" "${STARTED_AT}" "${FINISHED_AT}" "${ACTUAL_MOEI_COMMIT}" "${PATCH_SHA256}" \
  "${EXTENSION}" "${EXTENSION_SHA256}" "${RUNTIME_ELF_SHA256}" \
  "${RUNTIME_FINGERPRINT_JSON_SHA256}" "${BUILD_LOG_SHA256}" <<'PY'
import json
import os
import sys

(path, returncode, started, finished, moe_commit, patch_sha, extension,
 extension_sha, runtime_elf_sha, runtime_manifest_sha, log_sha) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "status": "complete" if int(returncode) == 0 else "failed",
    "returncode": int(returncode),
    "started_at": started,
    "finished_at": finished,
    "moe_infinity_commit": moe_commit,
    "moe_infinity_patch_sha256": patch_sha,
    "extension": os.path.realpath(extension),
    "extension_sha256": extension_sha,
    "runtime_elf_sha256": runtime_elf_sha,
    "runtime_elf_fingerprint": "runtime_elf_fingerprint.json",
    "runtime_elf_fingerprint_sha256": runtime_manifest_sha,
    "build_log": "build.log",
    "build_log_sha256": log_sha,
    "build_environment": {
        "MOE_BUILD_EXTENSIONS": "_store",
        "MOE_ENABLE_SM90": "0",
        "MOE_ENABLE_SM120": "1",
        "MAX_JOBS": "14",
        "NVTX_DISABLE": "0",
        "CUDA_HOME": "/usr/local/cuda-12.8",
        "NVTX_INCLUDE_DIR": "/usr/local/cuda-12.8/nsight-systems-2024.6.2/target-linux-x64/nvtx/include",
        "ninja_required": True,
    },
}
with open(path, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

(( RC == 0 )) || {
  tail -n 200 "${BUILD_LOG}" >&2 || true
  exit "${RC}"
}
echo "receipt=${OUTPUT_DIR}/store_build_receipt.json"
echo "extension_sha256=${EXTENSION_SHA256}"
echo "runtime_elf_sha256=${RUNTIME_ELF_SHA256}"
