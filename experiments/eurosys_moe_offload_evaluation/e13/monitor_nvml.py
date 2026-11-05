#!/usr/bin/env python3
"""Sample nvidia-smi metrics for one E13 cold-process arm."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 0.02:
        raise SystemExit("interval must be at least 0.02 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    fields = [
        "unix_s",
        "gpu_util_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_w",
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "pstate",
        "pcie_rx_kib_s",
        "pcie_tx_kib_s",
        "pcie_gen",
        "pcie_width",
        "sample_ok",
    ]

    def safe(call):
        try:
            return call()
        except Exception:
            return None

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    try:
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            deadline = time.monotonic()
            while not STOP:
                started = time.time()
                utilization = safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
                memory = safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
                row = {
                    "unix_s": started,
                    "gpu_util_percent": getattr(utilization, "gpu", None),
                    "memory_used_mib": memory.used / (1 << 20) if memory else None,
                    "memory_total_mib": memory.total / (1 << 20) if memory else None,
                    "power_w": safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0),
                    "temperature_c": safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                    "sm_clock_mhz": safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
                    "memory_clock_mhz": safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
                    "pstate": safe(lambda: pynvml.nvmlDeviceGetPerformanceState(handle)),
                    "pcie_rx_kib_s": safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)),
                    "pcie_tx_kib_s": safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)),
                    "pcie_gen": safe(lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)),
                    "pcie_width": safe(lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)),
                    "sample_ok": int(utilization is not None and memory is not None),
                }
                writer.writerow(row)
                stream.flush()
                deadline += args.interval
                time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
