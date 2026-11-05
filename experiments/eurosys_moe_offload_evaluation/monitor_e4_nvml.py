#!/usr/bin/env python3
"""Sample PCIe throughput and coarse GPU activity for E4."""

from __future__ import annotations

import argparse
import csv
import signal
import time
from pathlib import Path

try:
    import pynvml  # type: ignore[no-redef]
except ImportError:
    from vllm.third_party import pynvml  # type: ignore[no-redef]


STOP = False


def stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def safe(call, default=None):
    try:
        return call()
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-ms", type=int, default=200)
    args = parser.parse_args()
    if args.interval_ms < 20:
        raise ValueError("interval must be at least 20 ms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    fields = [
        "epoch_seconds", "monotonic_seconds", "pcie_rx_kib_s", "pcie_tx_kib_s",
        "gpu_util_pct", "memory_util_pct", "memory_used_mib", "power_w",
        "sm_clock_mhz", "pcie_gen", "pcie_width",
    ]
    try:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            stream.flush()
            deadline = time.monotonic()
            while not STOP:
                utilization = safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
                memory = safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
                row = {
                    "epoch_seconds": time.time(),
                    "monotonic_seconds": time.monotonic(),
                    "pcie_rx_kib_s": safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)),
                    "pcie_tx_kib_s": safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)),
                    "gpu_util_pct": getattr(utilization, "gpu", None),
                    "memory_util_pct": getattr(utilization, "memory", None),
                    "memory_used_mib": (
                        memory.used / (1 << 20) if memory is not None else None
                    ),
                    "power_w": safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0),
                    "sm_clock_mhz": safe(lambda: pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_SM)),
                    "pcie_gen": safe(lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)),
                    "pcie_width": safe(lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)),
                }
                writer.writerow(row)
                stream.flush()
                deadline += args.interval_ms / 1000.0
                time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
