// SPDX-License-Identifier: Apache-2.0
// Reuse the already-verified ioctl-90/91 ABI shim. This translation unit is
// intentionally tiny so the torch E2E probe and the slot microbenchmark use
// exactly the same existing-UVM-fd discovery and ioctl structures.

#include "../uvm_512x4m_slot_shared_push_compare/slot_batch_shim.cu"
