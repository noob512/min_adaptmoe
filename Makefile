PYTHON ?= workloads/vllm/vllm/.venv/bin/python
FIGURE_PYTHON ?= python3
OUTPUT ?= .build/prepared-inputs

.PHONY: check inputs driver allocator test figures
check:
	python3 scripts/check_source.py

inputs:
	PREPARE_ONLY=1 VLLM_PY=/usr/bin/python3 bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh "$(OUTPUT)" 1 20

driver:
	bash scripts/build_driver.sh

allocator:
	$(MAKE) -C workloads/vllm/vllm/uvm_test

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q -p no:cacheprovider tests

figures:
	$(FIGURE_PYTHON) evaluation/render_figures.py --output .build/figures
