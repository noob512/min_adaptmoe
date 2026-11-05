#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MOEI_ROOT="${REPO_ROOT}/MoE-Infinity"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python}"
QUALIFIER="${SCRIPT_DIR}/qualify_moe_infinity.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_moe_infinity_qualification.py"

EXPECTED_MOEI_COMMIT="9741f80b53374f0d98e7bd8f4a777b816bf68de8"
EXPECTED_MOEI_PATCH_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
REVIEWED_STORE_SHA256="b0b2b0e3ed3f80839039c33cccf38454aba14b0d6579962c379f064ee60b905b"
EXPECTED_STORE_RUNTIME_SHA256="b29484249a6a33402515e66e835fb3c4bdaa6b8fab0d87e82d2f1e456fd27b21"
STORE_BUILD_RECEIPT="${STORE_BUILD_RECEIPT:-}"
ELF_FINGERPRINTER="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/elf_runtime_fingerprint.py"
EXPECTED_MODEL_SNAPSHOT="85864749cd611b4353ce1decdb286193298f64c7"
EVALUATION_PROMPT_FIXTURE="${SCRIPT_DIR}/fixtures/moei_deepseek_v2_lite_evaluation_prompt512_v1.json"
EXPECTED_EVALUATION_FIXTURE_SHA256="5315aa367325a0adea8ac30fc7d16659bd0bdcbcfe608f8f5c3d59ea7d415bd5"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/${EXPECTED_MODEL_SNAPSHOT}}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/moe_infinity_demand_qualification_$(date +%Y%m%d_%H%M%S)}"
# The DeepSeek-V2-Lite expert store is roughly 30 GB.  Keep it out of the
# repository and out of /tmp; only the compact, immutable run evidence belongs
# under experiments/broad_offload_compare/artifacts.
OFFLOAD_DIR="${OFFLOAD_DIR:-/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_${EXPECTED_MODEL_SNAPSHOT}}"
PROMPT_LENGTH="${PROMPT_LENGTH:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"
REPETITIONS="${REPETITIONS:-5}"
DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO:-0.75}"
NUM_THREADS="${NUM_THREADS:-1}"
CPUSET="${CPUSET:-0-13}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}"
EVENT_SAMPLE_LIMIT="${EVENT_SAMPLE_LIMIT:-256}"
PROMPT_TOKEN_IDS_FILE="${PROMPT_TOKEN_IDS_FILE:-${EVALUATION_PROMPT_FIXTURE}}"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

canonical_path() {
  readlink -m -- "$1"
}

require_persistent_demand_paths() {
  local run_path offload_path repo_path
  run_path="$(canonical_path "${RUN_ROOT}")"
  offload_path="$(canonical_path "${OFFLOAD_DIR}")"
  repo_path="$(canonical_path "${REPO_ROOT}")"
  case "${run_path}" in
    /tmp|/tmp/*) die "demand qualification RUN_ROOT must not be under /tmp: ${run_path}" ;;
  esac
  case "${offload_path}" in
    /tmp|/tmp/*) die "OFFLOAD_DIR must not be under /tmp: ${offload_path}" ;;
  esac
  case "${offload_path}" in
    "${repo_path}"|"${repo_path}"/*)
      die "the ~30 GB OFFLOAD_DIR must be outside the repository: ${offload_path}"
      ;;
  esac
}

nearest_existing_parent() {
  local path
  path="$(canonical_path "$1")"
  while [[ ! -e "${path}" ]]; do
    path="$(dirname "${path}")"
  done
  printf '%s\n' "${path}"
}

prompt_fixture_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

validate_prompt_fixture() {
  local path="$1" expected_role="${2:-}"
  require_file "${path}"
  "${PYTHON_BIN}" - "${path}" "${PROMPT_LENGTH}" \
    "${EXPECTED_MODEL_SNAPSHOT}" "${expected_role}" <<'PY'
import hashlib
import json
import sys

path, expected_length, expected_snapshot, expected_role = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if not isinstance(payload, dict):
    raise SystemExit("frozen prompt fixture must be a JSON object")
token_ids = payload.get("prompt_token_ids")
if not isinstance(token_ids, list) or any(
    isinstance(value, bool) or not isinstance(value, int) or value < 0
    for value in token_ids
):
    raise SystemExit("fixture prompt_token_ids must be non-negative integers")
if len(token_ids) != int(expected_length):
    raise SystemExit(
        f"fixture has {len(token_ids)} IDs, expected {expected_length}"
    )
if payload.get("prompt_length") != int(expected_length):
    raise SystemExit("fixture prompt_length metadata mismatch")
if payload.get("model_snapshot_name") != expected_snapshot:
    raise SystemExit("fixture model snapshot metadata mismatch")
if expected_role and payload.get("fixture_role") != expected_role:
    raise SystemExit(
        f"fixture role={payload.get('fixture_role')!r}, expected {expected_role!r}"
    )
encoded = ",".join(str(value) for value in token_ids).encode("ascii")
token_hash = hashlib.sha256(encoded).hexdigest()
if payload.get("prompt_token_ids_sha256") != token_hash:
    raise SystemExit("fixture embedded token-ID SHA-256 mismatch")
print(
    f"prompt_fixture={path} role={payload.get('fixture_role')} "
    f"tokens={len(token_ids)} token_sha256={token_hash}"
)
PY
}

static_preflight() {
  require_dir "${MOEI_ROOT}"
  require_file "${PYTHON_BIN}"
  require_file "${QUALIFIER}"
  require_file "${SUMMARIZER}"
  require_file "${ELF_FINGERPRINTER}"
  [[ -n "${STORE_BUILD_RECEIPT}" ]] \
    || die "STORE_BUILD_RECEIPT must point to a fresh extension build receipt"
  require_file "${STORE_BUILD_RECEIPT}"
  require_dir "${MODEL}"
  require_file "${MODEL}/config.json"
  require_file "${MODEL}/model.safetensors.index.json"
  validate_prompt_fixture "${PROMPT_TOKEN_IDS_FILE}"
  command -v git >/dev/null || die "git is unavailable"
  command -v readlink >/dev/null || die "readlink is unavailable"
  command -v df >/dev/null || die "df is unavailable"

  local commit snapshot extension patch_sha256 extension_sha256
  local extension_runtime_sha256 model_bytes model_gib
  local offload_parent available_bytes available_gib
  commit="$(git -C "${MOEI_ROOT}" rev-parse HEAD)"
  snapshot="$(basename "${MODEL}")"
  extension="$(find "${MOEI_ROOT}/moe_infinity" -maxdepth 1 -type f -name '_store*.so' -print -quit)"
  [[ "${commit}" == "${EXPECTED_MOEI_COMMIT}" ]] \
    || die "MoE-Infinity commit drift: ${commit}"
  [[ "${snapshot}" == "${EXPECTED_MODEL_SNAPSHOT}" ]] \
    || die "model snapshot drift: ${snapshot}"
  [[ -n "${extension}" ]] || die "compiled moe_infinity._store extension is absent"
  patch_sha256="$(git -C "${MOEI_ROOT}" diff --binary | sha256sum | awk '{print $1}')"
  [[ "${patch_sha256}" == "${EXPECTED_MOEI_PATCH_SHA256}" ]] \
    || die "MoE-Infinity compatibility patch drift: ${patch_sha256}"
  extension_sha256="$(sha256sum "${extension}" | awk '{print $1}')"
  extension_runtime_sha256="$("${PYTHON_BIN}" "${ELF_FINGERPRINTER}" "${extension}")"
  [[ "${extension_runtime_sha256}" == "${EXPECTED_STORE_RUNTIME_SHA256}" ]] \
    || die "compiled _store runtime image does not match the reviewed build: ${extension_runtime_sha256}"
  "${PYTHON_BIN}" - "${STORE_BUILD_RECEIPT}" "${EXPECTED_MOEI_COMMIT}" \
    "${EXPECTED_MOEI_PATCH_SHA256}" "${EXPECTED_STORE_RUNTIME_SHA256}" \
    "${extension_sha256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
expected = {
    "status": "complete",
    "moe_infinity_commit": sys.argv[2],
    "moe_infinity_patch_sha256": sys.argv[3],
    "runtime_elf_sha256": sys.argv[4],
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(
            f"store build receipt {key}={receipt.get(key)!r}, expected {value!r}"
        )
if receipt.get("extension_sha256") != sys.argv[5]:
    raise SystemExit("store build receipt does not describe the active extension")
log_path = receipt_path.parent / str(receipt.get("build_log", ""))
if not log_path.is_file():
    raise SystemExit(f"store build log is absent: {log_path}")
digest = hashlib.sha256(log_path.read_bytes()).hexdigest()
if receipt.get("build_log_sha256") != digest:
    raise SystemExit("store build log SHA-256 mismatch")
runtime_path = receipt_path.parent / str(receipt.get("runtime_elf_fingerprint", ""))
if not runtime_path.is_file():
    raise SystemExit(f"runtime ELF fingerprint manifest is absent: {runtime_path}")
runtime_digest = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
if receipt.get("runtime_elf_fingerprint_sha256") != runtime_digest:
    raise SystemExit("runtime ELF fingerprint manifest SHA-256 mismatch")
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
if runtime.get("runtime_elf_sha256") != sys.argv[4]:
    raise SystemExit("runtime ELF fingerprint value mismatch")
if runtime.get("file_sha256") != sys.argv[5]:
    raise SystemExit("runtime ELF manifest does not describe the active extension")
print(
    f"store_build_receipt={receipt_path} build_log_sha256={digest} "
    f"runtime_elf_sha256={sys.argv[4]}"
)
PY
  require_persistent_demand_paths

  model_bytes="$("${PYTHON_BIN}" - "${MODEL}/model.safetensors.index.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get("metadata", {}).get("total_size")
if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise SystemExit("model index has no positive integer metadata.total_size")
print(value)
PY
)"
  model_gib="$("${PYTHON_BIN}" - "${model_bytes}" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 2**30:.3f}")
PY
)"
  offload_parent="$(nearest_existing_parent "${OFFLOAD_DIR}")"
  available_bytes="$(df -B1 --output=avail "${offload_parent}" | tail -n 1 | tr -d ' ')"
  if [[ ! -f "${OFFLOAD_DIR}/archer_index" ]] \
    && (( available_bytes < model_bytes + 5 * 1024 * 1024 * 1024 )); then
    die "insufficient free space for a new offload store: available=${available_bytes}, model=${model_bytes}, required_headroom=5GiB"
  fi
  available_gib="$("${PYTHON_BIN}" - "${available_bytes}" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 2**30:.3f}")
PY
)"
  printf '%s\n' \
    "MoE-Infinity static preflight: OK" \
    "  commit=${commit}" \
    "  model_snapshot=${snapshot}" \
    "  model_weight_bytes=${model_bytes} (${model_gib} GiB)" \
    "  compatibility_patch_sha256=${patch_sha256}" \
    "  compiled_extension=${extension}" \
    "  compiled_extension_sha256=${extension_sha256}" \
    "  compiled_extension_runtime_sha256=${extension_runtime_sha256}" \
    "  reviewed_legacy_extension_sha256=${REVIEWED_STORE_SHA256}" \
    "  store_build_receipt=$(canonical_path "${STORE_BUILD_RECEIPT}")" \
    "  prompt_fixture=$(canonical_path "${PROMPT_TOKEN_IDS_FILE}")" \
    "  prompt_fixture_sha256=$(prompt_fixture_sha256 "${PROMPT_TOKEN_IDS_FILE}")" \
    "  run_root=$(canonical_path "${RUN_ROOT}")" \
    "  offload_dir=$(canonical_path "${OFFLOAD_DIR}")" \
    "  offload_filesystem_available=${available_bytes} bytes (${available_gib} GiB)"
}

wait_for_clean_gpu() {
  local used_mib="" compute_processes="" _
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
    compute_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
      && (( used_mib <= GPU_CLEAN_MIB )) \
      && [[ -z "${compute_processes//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  nvidia-smi >&2 || true
  die "GPU did not become clean (used=${used_mib:-unknown} MiB, pids=${compute_processes:-none})"
}

preflight() {
  static_preflight
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v timeout >/dev/null || die "timeout is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"

  PYTHONPATH="${MOEI_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import torch
import transformers
import moe_infinity
import moe_infinity._store

assert torch.__version__ == "2.9.0+cu128", torch.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__
print(
    "MoE-Infinity qualification preflight: "
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"store={moe_infinity._store.__file__}"
)
PY
  echo "MoE-Infinity qualification preflight: OK"
}

record_metadata() {
  mkdir -p "${RUN_ROOT}/metadata"
  git -C "${MOEI_ROOT}" rev-parse HEAD > "${RUN_ROOT}/metadata/moe_infinity.commit"
  git -C "${MOEI_ROOT}" status --porcelain=v1 > "${RUN_ROOT}/metadata/moe_infinity.status"
  git -C "${MOEI_ROOT}" diff --binary > "${RUN_ROOT}/metadata/moe_infinity.patch"
  sha256sum "${RUN_ROOT}/metadata/moe_infinity.patch" \
    > "${RUN_ROOT}/metadata/moe_infinity.patch.sha256"
  sha256sum "${MODEL}/config.json" > "${RUN_ROOT}/metadata/model_config.sha256"
  sha256sum "${MODEL}/model.safetensors.index.json" \
    > "${RUN_ROOT}/metadata/model_index.sha256"
  cp --reflink=auto "${PROMPT_TOKEN_IDS_FILE}" \
    "${RUN_ROOT}/metadata/prompt_fixture.json"
  sha256sum "${RUN_ROOT}/metadata/prompt_fixture.json" \
    > "${RUN_ROOT}/metadata/prompt_fixture.json.sha256"
  printf '%s\n' "$(canonical_path "${PROMPT_TOKEN_IDS_FILE}")" \
    > "${RUN_ROOT}/metadata/prompt_fixture.source"
  find "${MOEI_ROOT}/moe_infinity" -maxdepth 1 -type f -name '_store*.so' \
    -exec sha256sum {} + > "${RUN_ROOT}/metadata/store_extension.sha256"
  cp --reflink=auto "${STORE_BUILD_RECEIPT}" \
    "${RUN_ROOT}/metadata/store_build_receipt.json"
  sha256sum "${RUN_ROOT}/metadata/store_build_receipt.json" \
    > "${RUN_ROOT}/metadata/store_build_receipt.json.sha256"
  sha256sum "${BASH_SOURCE[0]}" "${QUALIFIER}" "${SUMMARIZER}" \
    "${ELF_FINGERPRINTER}" \
    "${REPO_ROOT}/docs/eurosys_moe_offload_evaluation_plan.md" \
    > "${RUN_ROOT}/metadata/experiment_scripts.sha256"
  git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_ROOT}/metadata/host_repo.commit"
  uname -a > "${RUN_ROOT}/metadata/uname.txt"
  {
    printf '# relative_path\tsize_bytes\tresolved_path\n'
    local model_file
    for model_file in "${MODEL}"/*.safetensors; do
      printf '%s\t%s\t%s\n' \
        "$(basename "${model_file}")" \
        "$(stat -Lc '%s' "${model_file}")" \
        "$(readlink -f "${model_file}")"
    done
  } > "${RUN_ROOT}/metadata/model_shards.tsv"
  sha256sum "${RUN_ROOT}/metadata/model_shards.tsv" \
    > "${RUN_ROOT}/metadata/model_shards.tsv.sha256"
  df -hT "$(nearest_existing_parent "${OFFLOAD_DIR}")" \
    > "${RUN_ROOT}/metadata/offload_filesystem.txt"
  {
    echo "run_root=$(canonical_path "${RUN_ROOT}")"
    echo "offload_dir=$(canonical_path "${OFFLOAD_DIR}")"
    echo "model=$(canonical_path "${MODEL}")"
    echo "prompt_length=${PROMPT_LENGTH}"
    echo "output_tokens=${OUTPUT_TOKENS}"
    echo "warmup_requests=${WARMUP_REQUESTS}"
    echo "repetitions=${REPETITIONS}"
    echo "device_memory_ratio=${DEVICE_MEMORY_RATIO}"
    echo "num_threads=${NUM_THREADS}"
    echo "cpuset=${CPUSET}"
    echo "prompt_token_ids_file=$(canonical_path "${PROMPT_TOKEN_IDS_FILE}")"
    echo "prompt_fixture_sha256=$(prompt_fixture_sha256 "${PROMPT_TOKEN_IDS_FILE}")"
  } > "${RUN_ROOT}/metadata/protocol.env"
  lscpu > "${RUN_ROOT}/metadata/lscpu.txt"
  nvidia-smi -q > "${RUN_ROOT}/metadata/nvidia_smi_q.txt"
  if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip freeze > "${RUN_ROOT}/metadata/pip_freeze.txt"
  elif command -v uv >/dev/null 2>&1; then
    uv pip freeze --python "${PYTHON_BIN}" \
      > "${RUN_ROOT}/metadata/pip_freeze.txt"
  else
    die "cannot freeze Python environment: neither pip nor uv is available"
  fi
}

snapshot_offload_dir() {
  local label="$1"
  local output="${RUN_ROOT}/metadata/offload_${label}.tsv"
  if [[ ! -d "${OFFLOAD_DIR}" ]]; then
    printf '# absent\t%s\n' "$(canonical_path "${OFFLOAD_DIR}")" > "${output}"
  else
    {
      printf '# path\t%s\n' "$(canonical_path "${OFFLOAD_DIR}")"
      printf '# apparent_bytes\t'
      du -sb "${OFFLOAD_DIR}" | awk '{print $1}'
      printf '# allocated_bytes\t'
      du -sB1 "${OFFLOAD_DIR}" | awk '{print $1}'
      printf '# relative_path\tsize_bytes\tmtime_epoch\n'
      find "${OFFLOAD_DIR}" -xdev -type f \
        -printf '%P\t%s\t%T@\n' | LC_ALL=C sort
    } > "${output}"
  fi
  sha256sum "${output}" > "${output}.sha256"
}

write_terminal_marker() {
  local terminal_status="$1" returncode="$2" stage="$3"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${terminal_status}" \
    "${returncode}" "${stage}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
status = sys.argv[2]
returncode = int(sys.argv[3])
stage = sys.argv[4]
if status not in {"complete", "failed"}:
    raise SystemExit(f"invalid terminal status: {status}")

def digest(path: Path):
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

result_path = root / "cases" / "demand-only" / "result.json"
snapshot_path = root / "moe_infinity_qualification_snapshot.json"
result = json.loads(result_path.read_text()) if result_path.is_file() else None
snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else None
if status == "complete":
    if returncode != 0:
        raise SystemExit("complete marker requires returncode=0")
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise SystemExit("complete marker requires result status=ok")
    if not isinstance(snapshot, dict) or snapshot.get("status") != "ok":
        raise SystemExit("complete marker requires snapshot status=ok")

payload = {
    "schema_version": 1,
    "experiment": "moe_infinity_public_demand_qualification",
    "status": status,
    "stage": stage,
    "returncode": returncode,
    "run_root": str(root.resolve()),
    "case_result_status": result.get("status") if isinstance(result, dict) else None,
    "snapshot_status": snapshot.get("status") if isinstance(snapshot, dict) else None,
    "evidence_sha256": {
        "result.json": digest(result_path),
        "qualification_snapshot.json": digest(snapshot_path),
        "qualification_report.md": digest(
            root / "MOE_INFINITY_QUALIFICATION_RESULTS.md"
        ),
        "offload_before.tsv": digest(root / "metadata" / "offload_before.tsv"),
        "offload_after.tsv": digest(root / "metadata" / "offload_after.tsv"),
        "prompt_fixture.json": digest(root / "metadata" / "prompt_fixture.json"),
    },
}
marker = root / ("COMPLETED.json" if status == "complete" else "FAILED.json")
if status == "complete":
    stale_failed = root / "FAILED.json"
    if stale_failed.is_file():
        attempts = root / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        archived = attempts / f"FAILED.before_complete.{os.getpid()}.json"
        os.replace(stale_failed, archived)
temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker)
print(f"terminal_marker={marker}")
PY
}

run_case() {
  local mode="$1"
  case "${mode}" in
    demand-only|upstream-speculative|fixed-speculative|overlap) ;;
    *) die "unknown mode: ${mode}" ;;
  esac
  preflight
  if [[ ! -f "${RUN_ROOT}/metadata/protocol.env" ]]; then
    record_metadata
  fi
  local case_dir="${RUN_ROOT}/cases/${mode}"
  local -a prompt_args=()
  if [[ -n "${PROMPT_TOKEN_IDS_FILE}" ]]; then
    require_file "${PROMPT_TOKEN_IDS_FILE}"
    prompt_args=(--prompt-token-ids-file "${PROMPT_TOKEN_IDS_FILE}")
  fi
  [[ ! -e "${case_dir}" ]] || die "case already exists: ${case_dir}"
  mkdir -p "${case_dir}" "${OFFLOAD_DIR}"
  wait_for_clean_gpu

  {
    echo "PYTHONPATH=${MOEI_ROOT}:${PYTHONPATH:-}"
    echo "HF_HUB_OFFLINE=1"
    echo "TRANSFORMERS_OFFLINE=1"
    echo "TOKENIZERS_PARALLELISM=false"
    echo "CUDA_VISIBLE_DEVICES=0"
    echo "OMP_NUM_THREADS=${NUM_THREADS}"
    echo "CPUSET=${CPUSET}"
  } > "${case_dir}/environment.txt"
  printf '%q ' \
    taskset -c "${CPUSET}" "${PYTHON_BIN}" "${QUALIFIER}" \
    --model "${MODEL}" \
    --offload-dir "${OFFLOAD_DIR}" \
    --output "${case_dir}/result.json" \
    --mode "${mode}" \
    --prompt-length "${PROMPT_LENGTH}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --repetitions "${REPETITIONS}" \
    --device-memory-ratio "${DEVICE_MEMORY_RATIO}" \
    --num-threads "${NUM_THREADS}" \
    --event-sample-limit "${EVENT_SAMPLE_LIMIT}" \
    "${prompt_args[@]}" \
    > "${case_dir}/command.txt"
  printf '\n' >> "${case_dir}/command.txt"

  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader,nounits -lms 200 \
    > "${case_dir}/nvml.csv" 2> "${case_dir}/nvml.err" &
  local monitor_pid=$!
  set +e
  env \
    PYTHONPATH="${MOEI_ROOT}:${PYTHONPATH:-}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS="${NUM_THREADS}" \
    timeout --signal=INT --kill-after=30s "${TIMEOUT_SECONDS}s" \
    taskset -c "${CPUSET}" "${PYTHON_BIN}" "${QUALIFIER}" \
      --model "${MODEL}" \
      --offload-dir "${OFFLOAD_DIR}" \
      --output "${case_dir}/result.json" \
      --mode "${mode}" \
      --prompt-length "${PROMPT_LENGTH}" \
      --output-tokens "${OUTPUT_TOKENS}" \
      --warmup-requests "${WARMUP_REQUESTS}" \
      --repetitions "${REPETITIONS}" \
      --device-memory-ratio "${DEVICE_MEMORY_RATIO}" \
      --num-threads "${NUM_THREADS}" \
      --event-sample-limit "${EVENT_SAMPLE_LIMIT}" \
      "${prompt_args[@]}" \
      > "${case_dir}/stdout.log" 2> "${case_dir}/stderr.log"
  local rc=$?
  set -e
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  printf '%s\n' "${rc}" > "${case_dir}/returncode.txt"
  if [[ "${rc}" -ne 0 ]]; then
    tail -n 160 "${case_dir}/stderr.log" >&2 || true
    tail -n 160 "${case_dir}/stdout.log" >&2 || true
    return "${rc}"
  fi
  wait_for_clean_gpu
  echo "completed=${case_dir}"
}

summarize() {
  local modes="${1:-demand-only,upstream-speculative,fixed-speculative,overlap}"
  "${PYTHON_BIN}" "${SUMMARIZER}" "${RUN_ROOT}" \
    --modes "${modes}" \
    --expected-prompt-tokens "${PROMPT_LENGTH}" \
    --expected-output-tokens "${OUTPUT_TOKENS}" \
    --expected-warmup-requests "${WARMUP_REQUESTS}" \
    --expected-repetitions "${REPETITIONS}" \
    --expected-num-threads "${NUM_THREADS}" \
    --expected-prompt-fixture-sha256 "$(prompt_fixture_sha256 "${PROMPT_TOKEN_IDS_FILE}")" \
    --output-json "${RUN_ROOT}/moe_infinity_qualification_snapshot.json" \
    --output-md "${RUN_ROOT}/MOE_INFINITY_QUALIFICATION_RESULTS.md" \
    > "${RUN_ROOT}/summarizer.stdout.json"
  echo "snapshot=${RUN_ROOT}/moe_infinity_qualification_snapshot.json"
  echo "report=${RUN_ROOT}/MOE_INFINITY_QUALIFICATION_RESULTS.md"
}

validate_demand_configuration() {
  [[ "${PROMPT_LENGTH}" == "512" ]] \
    || die "demand qualification requires PROMPT_LENGTH=512"
  [[ "${OUTPUT_TOKENS}" == "32" ]] \
    || die "demand qualification requires OUTPUT_TOKENS=32"
  [[ "${WARMUP_REQUESTS}" == "3" ]] \
    || die "demand qualification requires WARMUP_REQUESTS=3"
  [[ "${REPETITIONS}" == "5" ]] \
    || die "demand qualification requires REPETITIONS=5"
  [[ "${NUM_THREADS}" == "1" ]] \
    || die "demand qualification requires NUM_THREADS=1"
  [[ "$(canonical_path "${PROMPT_TOKEN_IDS_FILE}")" == \
      "$(canonical_path "${EVALUATION_PROMPT_FIXTURE}")" ]] \
    || die "demand qualification requires the preregistered evaluation fixture"
  [[ "$(prompt_fixture_sha256 "${PROMPT_TOKEN_IDS_FILE}")" == \
      "${EXPECTED_EVALUATION_FIXTURE_SHA256}" ]] \
    || die "evaluation prompt fixture SHA-256 drift"
  validate_prompt_fixture "${PROMPT_TOKEN_IDS_FILE}" evaluation
}

verify_recorded_protocol() {
  local protocol="${RUN_ROOT}/metadata/protocol.env"
  require_file "${protocol}"
  local -a expected_lines=(
    "run_root=$(canonical_path "${RUN_ROOT}")"
    "offload_dir=$(canonical_path "${OFFLOAD_DIR}")"
    "model=$(canonical_path "${MODEL}")"
    "prompt_length=${PROMPT_LENGTH}"
    "output_tokens=${OUTPUT_TOKENS}"
    "warmup_requests=${WARMUP_REQUESTS}"
    "repetitions=${REPETITIONS}"
    "device_memory_ratio=${DEVICE_MEMORY_RATIO}"
    "num_threads=${NUM_THREADS}"
    "cpuset=${CPUSET}"
    "prompt_token_ids_file=$(canonical_path "${PROMPT_TOKEN_IDS_FILE}")"
    "prompt_fixture_sha256=$(prompt_fixture_sha256 "${PROMPT_TOKEN_IDS_FILE}")"
  )
  local line
  for line in "${expected_lines[@]}"; do
    grep -Fqx -- "${line}" "${protocol}" \
      || die "recorded protocol differs from resume request: ${line}"
  done
}

verify_resume_provenance() {
  verify_recorded_protocol
  [[ "$(<"${RUN_ROOT}/metadata/moe_infinity.commit")" == \
      "${EXPECTED_MOEI_COMMIT}" ]] \
    || die "recorded MoE-Infinity commit mismatch"
  [[ "$(git -C "${MOEI_ROOT}" status --porcelain=v1)" == \
      "$(<"${RUN_ROOT}/metadata/moe_infinity.status")" ]] \
    || die "MoE-Infinity dirty status changed; use a new RUN_ROOT"
  local current_patch_sha recorded_patch_sha
  current_patch_sha="$(git -C "${MOEI_ROOT}" diff --binary | sha256sum | awk '{print $1}')"
  recorded_patch_sha="$(awk '{print $1}' "${RUN_ROOT}/metadata/moe_infinity.patch.sha256")"
  [[ "${current_patch_sha}" == "${recorded_patch_sha}" ]] \
    || die "MoE-Infinity patch changed; use a new RUN_ROOT"
  sha256sum --status -c "${RUN_ROOT}/metadata/experiment_scripts.sha256" \
    || die "qualification scripts changed; use a new RUN_ROOT"
  sha256sum --status -c "${RUN_ROOT}/metadata/store_extension.sha256" \
    || die "compiled extension changed; use a new RUN_ROOT"
  sha256sum --status -c "${RUN_ROOT}/metadata/store_build_receipt.json.sha256" \
    || die "recorded store build receipt changed; use a new RUN_ROOT"
  [[ "$(sha256sum "${STORE_BUILD_RECEIPT}" | awk '{print $1}')" == \
      "$(awk '{print $1}' "${RUN_ROOT}/metadata/store_build_receipt.json.sha256")" ]] \
    || die "current store build receipt differs from the recorded receipt"
  sha256sum --status -c "${RUN_ROOT}/metadata/model_config.sha256" \
    || die "model config changed; use a new RUN_ROOT"
  sha256sum --status -c "${RUN_ROOT}/metadata/model_index.sha256" \
    || die "model index changed; use a new RUN_ROOT"
  sha256sum --status -c "${RUN_ROOT}/metadata/prompt_fixture.json.sha256" \
    || die "recorded prompt fixture changed; use a new RUN_ROOT"
}

initialize_or_verify_metadata() {
  if [[ -f "${RUN_ROOT}/metadata/protocol.env" ]]; then
    verify_resume_provenance
  else
    record_metadata
  fi
}

demand_case_is_complete() {
  local case_dir="${RUN_ROOT}/cases/demand-only"
  [[ -f "${case_dir}/returncode.txt" ]] || return 1
  [[ "$(<"${case_dir}/returncode.txt")" == "0" ]] || return 1
  [[ -f "${case_dir}/result.json" ]] || return 1
  "${PYTHON_BIN}" - "${case_dir}/result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
if value.get("status") != "ok":
    raise SystemExit(1)
if value.get("qualification", {}).get("status") != "ok":
    raise SystemExit(1)
PY
}

archive_incomplete_demand_attempt() {
  local stamp archive_dir case_dir
  stamp="$(date +%Y%m%d_%H%M%S)_$$"
  archive_dir="${RUN_ROOT}/attempts/demand-only_${stamp}"
  case_dir="${RUN_ROOT}/cases/demand-only"
  mkdir -p "${archive_dir}"
  [[ ! -e "${case_dir}" ]] || mv "${case_dir}" "${archive_dir}/case"
  local path
  for path in \
    "${RUN_ROOT}/metadata/offload_before.tsv" \
    "${RUN_ROOT}/metadata/offload_before.tsv.sha256" \
    "${RUN_ROOT}/metadata/offload_after.tsv" \
    "${RUN_ROOT}/metadata/offload_after.tsv.sha256" \
    "${RUN_ROOT}/moe_infinity_qualification_snapshot.json" \
    "${RUN_ROOT}/MOE_INFINITY_QUALIFICATION_RESULTS.md" \
    "${RUN_ROOT}/summarizer.stdout.json" \
    "${RUN_ROOT}/COMPLETED.json" \
    "${RUN_ROOT}/FAILED.json"; do
    [[ ! -e "${path}" ]] || mv "${path}" "${archive_dir}/"
  done
  echo "archived_incomplete_attempt=${archive_dir}"
}

run_demand_attempt() {
  snapshot_offload_dir before
  set +e
  run_case demand-only
  local rc=$?
  set -e
  snapshot_offload_dir after
  if [[ "${rc}" -ne 0 ]]; then
    write_terminal_marker failed "${rc}" run-case
    wait_for_clean_gpu
    return "${rc}"
  fi
  set +e
  summarize demand-only
  local summary_rc=$?
  set -e
  if [[ "${summary_rc}" -ne 0 ]]; then
    write_terminal_marker failed "${summary_rc}" summarize
    return "${summary_rc}"
  fi
  write_terminal_marker complete 0 summarize
}

demand_qualification() {
  validate_demand_configuration
  [[ ! -e "${RUN_ROOT}/cases" ]] || die "RUN_ROOT already contains cases"
  require_persistent_demand_paths
  preflight
  initialize_or_verify_metadata
  run_demand_attempt
}

resume_demand_qualification() {
  validate_demand_configuration
  require_persistent_demand_paths
  preflight
  initialize_or_verify_metadata
  if [[ -e "${RUN_ROOT}/cases/demand-only" ]]; then
    if demand_case_is_complete; then
      if [[ ! -f "${RUN_ROOT}/metadata/offload_after.tsv" ]]; then
        snapshot_offload_dir after
      fi
      set +e
      summarize demand-only
      local summary_rc=$?
      set -e
      if [[ "${summary_rc}" -ne 0 ]]; then
        write_terminal_marker failed "${summary_rc}" summarize
        return "${summary_rc}"
      fi
      write_terminal_marker complete 0 summarize
      return 0
    fi
    archive_incomplete_demand_attempt
  fi
  mkdir -p "${RUN_ROOT}/cases"
  run_demand_attempt
}

sweep() {
  [[ ! -e "${RUN_ROOT}/cases" ]] || die "RUN_ROOT already contains cases"
  preflight
  record_metadata
  # Each mode is a separate cold process. Demand brackets the experimental
  # profiles only in the later formal ABBA comparison, not in qualification.
  run_case demand-only
  run_case upstream-speculative
  run_case fixed-speculative
  run_case overlap
  summarize demand-only,upstream-speculative,fixed-speculative,overlap
}

show_help() {
  cat <<'EOF'
Usage:
  run_moe_infinity_qualification.sh static-preflight
  run_moe_infinity_qualification.sh preflight
  run_moe_infinity_qualification.sh demand-qualification
  run_moe_infinity_qualification.sh resume-demand-qualification
  run_moe_infinity_qualification.sh run-case MODE
  run_moe_infinity_qualification.sh sweep
  run_moe_infinity_qualification.sh summarize

MODE is one of demand-only, upstream-speculative, fixed-speculative, overlap.
The persistent demand qualification uses one fresh process, batch=1, an exact
512-token prompt, 32 generated tokens, 3 warmups, 5 measured repetitions, one
expert-compute thread, and the HuggingFace streamer path.
EOF
}

case "${COMMAND}" in
  static-preflight)
    static_preflight
    ;;
  preflight)
    preflight
    ;;
  demand-qualification)
    [[ $# -eq 0 ]] || die "demand-qualification takes no arguments"
    demand_qualification
    ;;
  resume-demand-qualification)
    [[ $# -eq 0 ]] || die "resume-demand-qualification takes no arguments"
    resume_demand_qualification
    ;;
  run-case)
    [[ $# -eq 1 ]] || die "run-case requires one MODE"
    run_case "$1"
    ;;
  sweep)
    [[ $# -eq 0 ]] || die "sweep takes no arguments"
    sweep
    ;;
  summarize)
    summarize
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
