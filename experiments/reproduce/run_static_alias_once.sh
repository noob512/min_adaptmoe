#!/usr/bin/env bash
set -euo pipefail

# Run one small original-VA static-alias CUDA Graph replay.  The model is not
# needed: this exercises the UVM mapping, Graph transport and teardown path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROBE="${REPO_ROOT}/experiments/uvm_static_alias_original_va/probe.py"
MODULE="${CUSTOM_UVM:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
PYTHON="${PYTHON:-${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python}"
OUTPUT="${1:-${REPO_ROOT}/experiments/reproduce/artifacts/static-alias-$(date -u +%Y%m%dT%H%M%SZ).json}"
REPLAYS="${REPLAYS:-1}"
SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-67108864}"
ROUTE_LOG_OUTPUT="${ROUTE_LOG_OUTPUT:-${OUTPUT%.json}.route.log}"

[[ -f "${PROBE}" ]] || { echo "missing probe: ${PROBE}" >&2; exit 1; }
[[ -f "${MODULE}" ]] || {
  echo "set CUSTOM_UVM to a compatible nvidia-uvm.ko" >&2
  exit 1
}
[[ -x "${PYTHON}" ]] || {
  echo "set PYTHON to a vLLM-compatible Python environment" >&2
  exit 1
}
[[ ! -e "${OUTPUT}" ]] || { echo "output already exists: ${OUTPUT}" >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT}")"

CUSTOM_UVM="${MODULE}" PYTHON="${PYTHON}" PROBE="${PROBE}" \
  OUTPUT="${OUTPUT}" REPLAYS="${REPLAYS}" \
  SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES}" \
  ROUTE_LOG_OUTPUT="${ROUTE_LOG_OUTPUT}" \
  bash "${REPO_ROOT}/experiments/uvm_static_alias_original_va/run_with_module_cycle.sh"

PYTHON="${PYTHON}" "${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {
    "ok": True,
    "remote_cpu_backing_restored": True,
    "sealed_policy_guards": {"all_rejected_in_use": True},
    "post_destroy_policy_recovery": {"all_accepted": True},
}
for key, expected in required.items():
    if isinstance(expected, dict):
        if any(result.get(key, {}).get(child) != value for child, value in expected.items()):
            raise SystemExit(f"reproduction gate failed: {key}")
    elif result.get(key) != expected:
        raise SystemExit(f"reproduction gate failed: {key}")
print(json.dumps({
    "status": "ok",
    "output": str(Path(sys.argv[1]).resolve()),
    "pte_mode": result.get("pte_mode"),
    "measured_replays": result.get("graph", {}).get("measured_replays"),
    "gpu_ms_per_replay": result.get("graph", {}).get("gpu_ms_per_replay"),
}, indent=2))
PY
