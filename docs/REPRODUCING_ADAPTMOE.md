# Build and reproduce

Run commands from the repository root. The main experiment compares four
mechanisms under mixed scheduling using a frozen ShareGPT trace.

## Environment and vLLM

The implementation was developed for a single RTX 5090 (32,607 MiB), NVIDIA
595.71.05, Linux 6.13.12 and Python 3.12. Matching NVIDIA base driver and UVM
ABI are required. Other GPUs, drivers and multi-GPU execution are not qualified
by this package. The modified vLLM build requires PyTorch 2.9.0 and a compatible
CUDA toolkit. Pinned extension dependencies are fetched by CMake; dependency
caches are intentionally excluded. On Ubuntu, install the Python venv/devel
packages (for example `python3.12-venv` and `python3-dev`) and build tools
before creating the environment.

```bash
cd workloads/vllm/vllm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/build.txt
VLLM_VERSION_OVERRIDE=0.0.0+adaptmoe python -m pip install --no-build-isolation -e .
python -m pip install pytest matplotlib
cd ../../..
workloads/vllm/vllm/.venv/bin/python -c 'import vllm; print(vllm.__file__)'
```

Use this source tree so the installed vLLM includes the AdaptMoE changes. The
version override also supports builds from a source archive.

## Driver and allocator

Install matching kernel headers and compiler tools. The driver helper obtains
the checksum-pinned official NVIDIA source, overlays the AdaptMoE UVM tree and
builds modules without installing or loading them:

```bash
bash scripts/build_driver.sh
make allocator
```

See [driver details](../kernel-module/README.md) for offline preparation,
`DRIVER_BUILD_DIR` and `SYSSRC`. Building all modules supplies the base-driver
symbol table needed by UVM. For the allocator, override `CUDA_HOME` and `NVCC`
if CUDA is elsewhere. Generated modules and libraries are not distributed.

The real replay wrapper needs sudo and exclusive access to the GPU: it checks
for clients, temporarily loads the custom UVM, and restores stock UVM and
previously active exporter services on exit. It cannot run concurrently with
other GPU services. No module change is made by the CPU checks or build helpers.

## Checks without model execution

```bash
make check
make inputs
# Full focused runtime tests, after installing this vLLM and its dependencies:
make test
# Small source/verifier checks requiring only pytest:
python3 -m pytest -q \
  experiments/vllm_uvm_prefill_alias_decode_remote_graph/test_verify_experiment.py \
  experiments/uvm_static_alias_original_va/test_kernel_error_guards.py
```

`make check` checks source syntax, portable paths, the frozen
request hash and nested expert plans. `make inputs` prepares QPS=1 and QPS=20
inputs under a new `.build/prepared-inputs` directory, without vLLM imports,
model weights, the complete dataset or GPU access. Choose a fresh `OUTPUT`
for another preparation. Prepared inputs are not performance results.

Focused tests cover access lifecycle, ioctl contracts, expert plan changes,
KV queues and capacity policy. They do not replace GPU correctness checks.

## Canonical four-arm replay

Provide DeepSeek-V2-Lite-Chat weights and the ShareGPT V3 source dataset.
The frozen fixture includes token IDs so all arms use identical request
payloads and order. The full dataset is used to check source provenance.

```bash
MODEL=/absolute/path/to/DeepSeek-V2-Lite-Chat \
SHAREGPT_DATASET=/absolute/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
ALLOW_GPU_RUN=1 SAMPLES=1 \
bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh \
  /absolute/new/artifacts/kv1-qps1 1
```

Supported offered rates are `1 2 4 8 10 20 30`; pass several as trailing
arguments to run a sweep. `SAMPLES=1` means one process per arm. `SAMPLES=2`
explicitly adds a second process per arm in reverse order. There is no hidden
repetition and no reuse of historical results. Every output directory must
be new. `VLLM_PY`, `CUSTOM_UVM`, `CPUSET` (default 0–13) and `TIMEOUT_SEC`
can be overridden to match the environment.

| Item | Protocol |
|---|---|
| Arms | Native UVA → Native Copy → AdaptMoE fixed K1 → AdaptMoE dynamic K1↔K2 |
| Requests | 128 distinct ShareGPT prompts; 1024 input / 16 output tokens |
| Arrival | Request index `i` has planned offset `i / QPS` |
| Model | DeepSeek-V2-Lite-Chat, BF16 |
| Scheduler | Mixed continuous batching; C128; 8192-token budget; max length 1088 |
| Attention / Graph | TRITON_MLA; eager; CUDA Graph `NONE` |
| Prefix cache / sampling | Disabled; greedy, ignore EOS |
| Access thresholds | PARK≤16; RESUME≥128; keep view between thresholds |
| Fixed KV | 2,155 usable blocks / 34,480 tokens, approximately 1 GiB |
| Dynamic KV | K1↔K2; up to 4,313 usable blocks / 69,008 tokens |
| Expert pairing | Adapt K1/cold8, K2/cold11; Native seven whole Parameters |

The AdaptMoE fixed arm also reserves a K2-sized virtual KV allocation while
limiting residency/allocatable blocks to K1. Thus a nominal 2 GiB allocation
argument in its command does not mean fixed K2 usable capacity. The generated
`capacity_certificate.json`, runtime settings and observed pool report resolve
this distinction. Native offloads 2.56% more main weight bytes than Adapt K1;
this granularity difference prevents a strict pure-transport attribution.

Fixture SHA-256:
`fed1428ffe8d2fe7a4608b626041f31b81f22071876e330b495e3db3b8376a43`.
The fixture includes all 128 frozen prompt strings and token-ID sequences.
ShareGPT V3 source SHA-256:
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.
The model snapshot used to prepare the fixture was
`deepseek-ai/DeepSeek-V2-Lite-Chat@85864749cd611b4353ce1decdb286193298f64c7`.

## Measurement and output checks

Actual QPS is requests completed divided by the interval from first planned
arrival to final completion, including drain. TTFT includes arrival/admission
queueing. TPOT measures subsequent token intervals. Real capacity transition
time is included. There is no additional full conditioning request before the
measured trace, so these numbers must not be labeled pure steady-state latency.

The output retains configuration and input hashes, execution status, per-arm
results, manifests, runtime/kernel logs, module/cleanup snapshots, and summary
tables. Inspect `REPORT.md`, `summary.csv`, `output_checks.csv` and
`verification.json`. The verifier checks request counts, input identity,
arrival protocol, offload bytes, Graph mode, access/capacity events, faults and
cleanup. It compares complete outputs and first tokens. `ok_with_output_drift`
means execution passed with numerical output differences; it is not a strict
equivalence result. One process per arm provides no confidence interval.

See [VALIDATION.md](VALIDATION.md) for source checks, build results and the
scope of the validation.
