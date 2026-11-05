# P3 prefill to custom remote-read decode

This experiment tests a phase-aware policy on the real
Qwen3-30B-A3B-FP8 vLLM path. All arms use the same frozen token-ID request and
the same sparse placement plan: 48 layers, 12 cold experts per layer, 9
512-KiB regions per expert (54 MiB per layer).

The three process-isolated arms are:

- `H` (`hybrid`): P3 for the one prefill model forward, then an audited,
  fail-closed cutover to the custom managed-UVA `remote_read` policy before
  the first decode model forward;
- `P` (`p3`): P3 for prefill and every decode forward;
- `U` (`remote`): custom managed-UVA `remote_read` for both phases.

Each request produces exactly eight tokens. With one unchunked request this
means exactly 48 registered-plan transitions for `H` and 384 transitions for
`P`. The hybrid result must report exactly 48 modules, 576 experts, and 5,184
regions remapped for remote access. `U` must establish the same 576-expert,
5,184-region remote placement without executing a P3 plan. All output-token
hashes must match.

The hybrid cutover uses the audited batch UAPI explicitly. Its 5,184 regions
are submitted in batches of at most 64 ranges, so the accepted result must
report exactly 81 owner-VA-space mapping ioctls and exactly 2,717,908,992
remotely mapped bytes. VA-space candidate probes are reported separately and
may increase the total ioctl count. A missing batch flag, a mapping-batch count
other than 81, partial byte count, or any mismatch in the
48-module/576-expert/5,184-region identity fails verification.

This is the phase-policy comparison. The separate
`experiments/vllm_prefill_p3_pinned_uva_compare` experiment provides the
prefill-only P3/pinned/UVA comparison. A sparse-12 pinned arm is not included
here because Qwen's packed expert parameters cannot be independently selected
by the existing vLLM pinned parameter offloader.

## Measurement contract

- one request and `max_num_seqs=1`;
- chunked prefill explicitly disabled;
- eager execution, no CUDA Graph;
- no warmup request in the measured process;
- no speculative decode, DBO, communication, or mixed prefill/decode batch;
- P3 still does not overlap layer computation and weight transfer.

The verifier reports TTFT (prefill), TPOT (decode), and request E2E medians.
The hybrid cutover synchronization, P3 close, remote-map, and total wall times
are reported separately so they are not mistaken for decode kernel time.
The runner explicitly sets `VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH=1` only
for `H`; `P` and `U` explicitly set it to zero to prevent inherited shell
state from changing the arm definition.

## VA-space and fd safety boundary

The cutover first requires the P3 abort audit to prove that the slot session
is cleared and has zero remaining slots and bytes. It then uses ioctl 84 with
the already-registered region IDs, rather than trusting raw virtual addresses.
Candidate initialized UVM fds are duplicated and probed until the fd owning
those region IDs is found; all mapping chunks then stay on that fd. Any partial
progress fixes the owner choice and fails closed instead of trying another
VA-space.

Action 21 installs missing CPU remote-READ PTEs and reports the final audited
state. Every result must show exact CPU residency, exact remote mapping, full
coverage, and zero unknown bytes. This intentionally accepts a PTE which was
already a valid CPU remote mapping after slot close, but never accepts a stale
slot mapping: the preceding abort contract proves all slot backing is gone.
Each duplicated fd is closed in a `finally` block.

This experiment remains intentionally restricted to one process, one GPU, and
one serving VA-space because the phase cutover policy itself has not been
qualified for multi-worker execution. The registered-region lookup nevertheless
removes enumeration order from VA-space selection: a wrong fd has no matching
IDs and cannot mutate the requested ranges.

## Static validation

This command does not initialize CUDA or load/unload a kernel module:

```bash
bash experiments/vllm_prefill_p3_decode_remote/run_compare.sh static-preflight
```

## Runtime

The runner intentionally contains no module-cycle logic. Load the P3 UVM
module externally with the same parameters used by the sparse P3 E2E test,
including copy-run/metadata caches, direct cutover, fast validation, shadow
PTE, dual-bank event, and CUDA fence. The slot budget must be at least
113,246,208 bytes.

Then run:

```bash
ALLOW_GPU_RUN=1 \
  bash experiments/vllm_prefill_p3_decode_remote/run_compare.sh run
```

The default balanced order is `HPUUPH`. `ORDER`, `RUN_ROOT`, `MODEL`,
`EVAL_FIXTURE`, `GPU_INDEX`, `CPUSET`, `CPU_THREADS`, and `TIMEOUT_SEC` can be
overridden. Every arm starts in a fresh process, and GPU 0 must be at exactly
0 MiB with no compute client before and after each arm.

The final `verification.json` accepts a hybrid run only if the P3 session is
closed once, the remote mapping audit succeeds, the phase-specific transition
counts are exact, the batch cutover reports 81 ioctls and 2,717,908,992 mapped
bytes (plus any separately reported fd probes), the pure remote arm never
enters P3, and all three arms produce
identical output-token hashes.
