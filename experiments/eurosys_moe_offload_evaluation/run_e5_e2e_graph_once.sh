#!/usr/bin/env bash
set -euo pipefail

E5_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E5_REPO_ROOT="$(cd "${E5_SCRIPT_DIR}/../.." && pwd)"
BASE_HARNESS="${E5_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

SCRIPT_DIR="${E5_SCRIPT_DIR}"
REPO_ROOT="${E5_REPO_ROOT}"
VERIFY="${SCRIPT_DIR}/verify_e5_e2e_graph_once.py"

generate_case_plan() {
  local output="$1" mode="$2" repeats="$3"
  "${VLLM_PY}" - "${output}" "${mode}" "${repeats}" <<'PY'
import csv, sys
path, mode, repeats = sys.argv[1:]
if mode != "formal" or int(repeats) != 1:
    raise SystemExit("E5 one-round runner requires formal mode and one repeat")
rows = []
for batch, symbols in ((1, "HG"), (4, "GH")):
    rows.extend(("performance", 1, batch, symbol) for symbol in symbols)
with open(path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(("ordinal", "kind", "sample", "batch", "symbol"))
    for ordinal, row in enumerate(rows, 1):
        writer.writerow((ordinal, *row))
PY
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  formal|run)
    export FORMAL_PERFORMANCE_REPEATS=1
    run_suite formal
    ;;
  *)
    echo "usage: $0 {static-preflight|formal}" >&2
    exit 2
    ;;
esac
