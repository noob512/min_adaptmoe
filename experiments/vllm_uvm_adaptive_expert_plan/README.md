# Adaptive expert plan control-plane experiment

This experiment exercises the production heat-counter methods on a CUDA GPU,
rotates the two heat buffers, verifies exact assignment accounting, applies the
CPU EWMA, and checks safe fixed-class selection.

```bash
cd workloads/vllm/vllm
PYTHONPATH=. .venv/bin/python \
  ../../../experiments/vllm_uvm_adaptive_expert_plan/run_gpu_control_plane.py
```

The experiment does not claim multi-class CUDA Graph replay qualification.
The current runtime can exchange experts within the active fixed-capacity
placement, but C2/C4/C6/C8 GraphExec capture and arbitrary slot rebinding are
still separate implementation work.

The Graph-class registry microexperiment captures two retained CUDA Graphs
over one stable GPU slot and switches between them without reset or recapture:

```bash
cd workloads/vllm/vllm
PYTHONPATH=. .venv/bin/python \
  ../../../experiments/vllm_uvm_adaptive_expert_plan/run_gpu_graph_class_registry.py
```

This proves the registry and stable-slot mechanism. The repository already has
four concrete C2/C4/C6/C8 plans; full-model qualification additionally requires
converting their monolithic per-plan arenas into retained stable slot envelopes.
