PYTHON ?= workloads/vllm/vllm/.venv/bin/python
OUTPUT ?= .build/prepared-inputs

.PHONY: check inputs driver allocator test
check:
	python3 scripts/check_source.py

inputs:
	PREPARE_ONLY=1 VLLM_PY=/usr/bin/python3 bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh "$(OUTPUT)" 1 20

driver:
	bash scripts/build_driver.sh

allocator:
	$(MAKE) -C workloads/vllm/vllm/uvm_test

test:
	$(PYTHON) -m pytest -q tests
