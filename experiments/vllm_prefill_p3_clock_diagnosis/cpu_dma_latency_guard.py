#!/usr/bin/env python3
"""Hold a Linux CPU DMA-latency QoS request without periodic wakeups.

The ``hold`` command opens ``/dev/cpu_dma_latency``, writes one native signed
32-bit latency value, and then blocks without periodic polling.  Keeping the
file descriptor open maintains the PM QoS request; closing it releases the
request.

This helper intentionally does not launch or inspect CUDA.  The device is
normally root-only, so a benchmark driver should start this helper with sudo,
wait for ``--ready-file``, run the benchmark as the ordinary user, and finally
send SIGTERM to the PID recorded in the ready file.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_DEVICE = Path("/dev/cpu_dma_latency")


def device_metadata(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "device": str(path.resolve()),
        "character_device": stat.S_ISCHR(info.st_mode),
        "mode_octal": oct(stat.S_IMODE(info.st_mode)),
        "owner_uid": info.st_uid,
        "owner_gid": info.st_gid,
        "caller_uid": os.getuid(),
        "caller_euid": os.geteuid(),
        "caller_can_read": os.access(path, os.R_OK),
        "caller_can_write": os.access(path, os.W_OK),
        "rdev_major": os.major(info.st_rdev),
        "rdev_minor": os.minor(info.st_rdev),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise RuntimeError(f"ready-file parent does not exist: {path.parent}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    descriptor = os.open(temporary, flags, 0o644)
    try:
        data = encoded.encode("utf-8")
        written = os.write(descriptor, data)
        if written != len(data):
            raise RuntimeError(f"short ready-file write: {written}/{len(data)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def inspect_device(args: argparse.Namespace) -> int:
    payload = {"status": "ok", **device_metadata(args.device)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def hold_request(args: argparse.Namespace) -> int:
    if args.latency_us < 0 or args.latency_us > 2_147_483_647:
        raise SystemExit("--latency-us must fit a non-negative signed 32-bit integer")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.ready_file.exists():
        raise SystemExit(f"ready file already exists: {args.ready_file}")

    metadata = device_metadata(args.device)
    if not metadata["character_device"]:
        raise SystemExit(f"not a character device: {args.device}")

    stop_reason = "signal"
    stopped = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_reason
        stop_reason = signal.Signals(signum).name
        stopped.set()

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)

    flags = os.O_RDWR | os.O_CLOEXEC
    descriptor = os.open(args.device, flags)
    acquired_epoch = time.time()
    try:
        payload = struct.pack("=i", args.latency_us)
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise RuntimeError(f"short cpu_dma_latency write: {written}/{len(payload)}")

        ready = {
            "status": "acquired",
            **metadata,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "latency_us": args.latency_us,
            "acquired_epoch": acquired_epoch,
            "timeout_seconds": args.timeout_seconds,
        }
        atomic_write_json(args.ready_file, ready)
        print(json.dumps(ready, sort_keys=True), flush=True)

        if not stopped.wait(args.timeout_seconds):
            stop_reason = "timeout"
    finally:
        os.close(descriptor)

    released = {
        "status": "released",
        "pid": os.getpid(),
        "latency_us": args.latency_us,
        "acquired_epoch": acquired_epoch,
        "released_epoch": time.time(),
        "reason": stop_reason,
    }
    print(json.dumps(released, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="report device metadata without opening the device"
    )
    inspect_parser.add_argument("--device", type=Path, default=DEFAULT_DEVICE)
    inspect_parser.set_defaults(function=inspect_device)

    hold_parser = subparsers.add_parser(
        "hold", help="hold a PM QoS latency request until signaled"
    )
    hold_parser.add_argument("--device", type=Path, default=DEFAULT_DEVICE)
    hold_parser.add_argument(
        "--latency-us",
        type=int,
        default=0,
        help="native signed 32-bit PM QoS request; use 0 for the strictest request",
    )
    hold_parser.add_argument("--ready-file", type=Path, required=True)
    hold_parser.add_argument("--timeout-seconds", type=int, default=7200)
    hold_parser.set_defaults(function=hold_request)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return int(args.function(args))
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
