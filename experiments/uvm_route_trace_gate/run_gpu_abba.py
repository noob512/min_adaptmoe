#!/usr/bin/env python3
"""Run an isolated OFF/ON route-trace ABBA around the fixed-bank GPU probe.

This controller never loads or unloads a kernel module. It requires the exact
stage-2 module to be active, toggles only uvm_phase_f_route_log_enable, launches
one fresh worker process per arm, and restores the original gate in a finally
block. Use --dry-run to inspect commands without touching sysfs or the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PROBE = (
    REPO_ROOT / "experiments/uvm_slot_transition_shared_64k_probe/run_probe.py"
)
DEFAULT_PYTHON = REPO_ROOT / "workloads/vllm/vllm/.venv/bin/python"
DEFAULT_ROUTE_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_enable"
)
DEFAULT_DELAY_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_delay_ms"
)
DEFAULT_ROUTE_STATS = Path(
    "/proc/driver/nvidia-uvm/phase_f_route_log_stats"
)
DEFAULT_SRCVERSION = Path("/sys/module/nvidia_uvm/srcversion")
FIXED_PARAMETERS = {
    "uvm_moe_slot_pte_remap_enable": 1,
    "uvm_moe_slot_transition_fast_index": 1,
    "uvm_moe_slot_transition_direct_cutover": 1,
}
ROUTE_ACTIVITY_COUNTERS = (
    "debug_emit_calls",
    "disabled_race_drops",
    "always_emit_calls",
    "timestamp_calls",
    "format_calls",
    "mutex_acquires",
    "records",
    "bytes",
)
ABBA = (False, True, True, False)
MIB = 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_scalar(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_int(path: Path) -> int:
    return int(read_scalar(path), 0)


def parse_route_stats(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        require(len(fields) == 2, f"route stats line {line_number} is malformed")
        result[fields[0]] = int(fields[1], 0)
    for name in ("enabled",) + ROUTE_ACTIVITY_COUNTERS:
        require(name in result, f"route stats is missing {name}")
    return result


def read_route_stats(path: Path) -> dict[str, int]:
    return parse_route_stats(path.read_text(encoding="utf-8"))


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    delta = {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in set(before) | set(after)
        if name != "enabled"
    }
    require(all(value >= 0 for value in delta.values()), "route counters moved backwards")
    return dict(sorted(delta.items()))


def set_route_gate(path: Path, enabled: int, *, dry_run: bool) -> None:
    require(enabled in (0, 1), "route gate must be zero or one")
    if dry_run:
        print(f"echo {enabled} | sudo tee {path}")
        return
    if os.access(path, os.W_OK):
        path.write_text(f"{enabled}\n", encoding="utf-8")
    else:
        completed = subprocess.run(
            ["sudo", "tee", str(path)],
            input=f"{enabled}\n",
            text=True,
            stdout=subprocess.DEVNULL,
            check=False,
        )
        require(completed.returncode == 0, f"failed to write route gate {path}")
    require(read_int(path) == enabled, "route gate write did not take effect")


def gpu_used_mib(device: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(device),
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def wait_for_zero_mib(device: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        used = gpu_used_mib(device)
        if used == 0:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU {device} still uses {used} MiB")
        time.sleep(0.25)


def validate_environment(args: argparse.Namespace) -> str:
    for path in (
        args.python,
        args.probe,
        args.route_parameter,
        args.delay_parameter,
        args.route_stats,
        args.srcversion,
    ):
        require(path.is_file(), f"missing prerequisite: {path}")

    loaded_srcversion = read_scalar(args.srcversion)
    require(
        loaded_srcversion == args.expected_srcversion,
        "loaded module srcversion does not match the pinned stage-2 build: "
        f"loaded={loaded_srcversion} expected={args.expected_srcversion}",
    )
    require(read_int(args.delay_parameter) == 0, "route log delay must remain zero")

    parameter_dir = args.route_parameter.parent
    for name, expected in FIXED_PARAMETERS.items():
        actual = read_int(parameter_dir / name)
        require(actual == expected, f"{name}={actual}, expected {expected}")

    budget = read_int(parameter_dir / "uvm_moe_slot_pool_vidmem_budget_bytes")
    required_budget = args.blocks * 2 * MIB
    require(
        budget >= required_budget,
        f"slot budget {budget} is below required {required_budget}",
    )
    return loaded_srcversion


def validate_route_delta(enabled: bool, delta: dict[str, int]) -> None:
    values = {name: int(delta.get(name, 0)) for name in ROUTE_ACTIVITY_COUNTERS}
    if not enabled:
        require(
            all(value == 0 for value in values.values()),
            f"OFF arm performed route-log work: {values}",
        )
        return

    debug_calls = values["debug_emit_calls"]
    require(debug_calls > 0, "ON arm emitted no debug route records")
    require(values["disabled_race_drops"] == 0, "route gate changed during an ON arm")
    require(values["always_emit_calls"] == 0, "probe unexpectedly took a fallback path")
    for name in ("timestamp_calls", "format_calls", "mutex_acquires", "records"):
        require(values[name] == debug_calls, f"{name} does not match debug emits")
    require(values["bytes"] > 0, "ON arm recorded zero route-log bytes")


def validate_probe_payload(
    payload: dict[str, Any],
    *,
    blocks: int,
    expected_srcversion: str,
) -> dict[str, int]:
    entries = blocks * 4
    logical_bytes = blocks * 2 * MIB
    require(payload.get("status") == "pass", f"GPU probe failed: {payload.get('error')}")
    require(payload.get("module_srcversion") == expected_srcversion, "module changed during arm")
    require(int(payload.get("blocks", -1)) == blocks, "probe block count changed")
    require(int(payload.get("entries", -1)) == entries, "probe entry count changed")
    require(int(payload.get("logical_bytes", -1)) == logical_bytes, "probe bytes changed")

    metrics: dict[str, int] = {}
    for field, expected_mode in (("restore_a", 21), ("transition_b", 20)):
        result = payload.get(field)
        require(isinstance(result, dict), f"probe omitted {field}")
        expected = {
            "ok": True,
            "mode": expected_mode,
            "requested_count": entries,
            "success_count": entries,
            "failure_count": 0,
            "requested_bytes": logical_bytes,
            "accepted_bytes": logical_bytes,
            "remap_count": entries,
            "ioctl_calls": 1,
        }
        for name, value in expected.items():
            require(result.get(name) == value, f"{field}.{name} changed")
        ioctl_wall_ns = int(result.get("ioctl_wall_ns", 0))
        require(ioctl_wall_ns > 0, f"{field} has no ioctl timing")
        metrics[f"{field}_ioctl_wall_ns"] = ioctl_wall_ns

    metrics["hot_ioctl_wall_ns"] = (
        metrics["restore_a_ioctl_wall_ns"]
        + metrics["transition_b_ioctl_wall_ns"]
    )
    return metrics


def write_report(summary: dict[str, Any], path: Path) -> None:
    comparison = summary["comparison"]
    lines = [
        "# Route trace GPU A/B result",
        "",
        f"- status: `{summary['status']}`",
        f"- module srcversion: `{summary['module_srcversion']}`",
        f"- blocks per layer: `{summary['blocks']}`",
        f"- entries per layer: `{summary['entries_per_layer']}`",
        f"- formal arms: `{len(summary['arms'])}`",
        "",
        "| metric | trace OFF median | trace ON median | ON/OFF |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("restore_a_ioctl_wall_ns", "mode21 restore ioctl"),
        ("transition_b_ioctl_wall_ns", "mode20 transition ioctl"),
        ("hot_ioctl_wall_ns", "combined hot ioctls"),
    ):
        item = comparison[name]
        lines.append(
            f"| {label} | {item['off_median_ns'] / 1e6:.3f} ms | "
            f"{item['on_median_ns'] / 1e6:.3f} ms | {item['on_over_off']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "OFF arms require zero delta for every route activity counter. ON arms "
            "require one timestamp, format, mutex acquisition, and record per debug emit. "
            "All arms passed the existing VA, checksum, bytes, PTE, and copy-push checks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def comparison(arms: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in (
        "restore_a_ioctl_wall_ns",
        "transition_b_ioctl_wall_ns",
        "hot_ioctl_wall_ns",
    ):
        off = [int(arm["metrics"][name]) for arm in arms if not arm["trace_enabled"]]
        on = [int(arm["metrics"][name]) for arm in arms if arm["trace_enabled"]]
        off_median = float(statistics.median(off))
        on_median = float(statistics.median(on))
        result[name] = {
            "off_median_ns": off_median,
            "on_median_ns": on_median,
            "on_over_off": on_median / off_median,
        }
    return result


def dry_run_commands(args: argparse.Namespace) -> int:
    for cycle in range(1, args.cycles + 1):
        for position, enabled in enumerate(ABBA, 1):
            label = "on" if enabled else "off"
            output = args.output_dir / f"cycle{cycle:02d}_{position:02d}_{label}.json"
            set_route_gate(args.route_parameter, int(enabled), dry_run=True)
            print(
                " ".join(
                    map(
                        str,
                        (
                            args.python,
                            args.probe,
                            "--blocks",
                            args.blocks,
                            "--device",
                            args.device,
                            "--expect-direct-cutover",
                            "--output",
                            output,
                        ),
                    )
                )
            )
    print(f"echo ORIGINAL_VALUE | sudo tee {args.route_parameter}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-srcversion", required=True)
    parser.add_argument("--blocks", type=int, default=288)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--route-parameter", type=Path, default=DEFAULT_ROUTE_PARAM)
    parser.add_argument("--delay-parameter", type=Path, default=DEFAULT_DELAY_PARAM)
    parser.add_argument("--route-stats", type=Path, default=DEFAULT_ROUTE_STATS)
    parser.add_argument("--srcversion", type=Path, default=DEFAULT_SRCVERSION)
    parser.add_argument("--zero-mib-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require(1 <= args.blocks <= 288, "blocks must be in [1, 288]")
    require(args.cycles >= 1, "cycles must be positive")

    if args.dry_run:
        return dry_run_commands(args)

    loaded_srcversion = validate_environment(args)
    original_gate = read_int(args.route_parameter)
    require(original_gate in (0, 1), "original route gate is not boolean")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arms: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "failed",
        "module_srcversion": loaded_srcversion,
        "blocks": args.blocks,
        "entries_per_layer": args.blocks * 4,
        "logical_bytes_per_layer": args.blocks * 2 * MIB,
        "cycles": args.cycles,
        "order_per_cycle": ["off", "on", "on", "off"],
        "original_route_gate": original_gate,
        "arms": arms,
    }
    failure: str | None = None

    try:
        for cycle in range(1, args.cycles + 1):
            for position, enabled in enumerate(ABBA, 1):
                wait_for_zero_mib(args.device, args.zero_mib_timeout)
                set_route_gate(args.route_parameter, int(enabled), dry_run=False)
                before = read_route_stats(args.route_stats)
                require(before["enabled"] == int(enabled), "stats gate disagrees with sysfs")

                label = "on" if enabled else "off"
                artifact = args.output_dir / f"cycle{cycle:02d}_{position:02d}_{label}.json"
                stdout_path = artifact.with_suffix(".stdout.txt")
                stderr_path = artifact.with_suffix(".stderr.txt")
                command = [
                    str(args.python),
                    str(args.probe),
                    "--blocks",
                    str(args.blocks),
                    "--device",
                    str(args.device),
                    "--expect-direct-cutover",
                    "--output",
                    str(artifact),
                ]
                with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr_file:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        check=False,
                    )
                after = read_route_stats(args.route_stats)
                delta = counter_delta(before, after)
                require(completed.returncode == 0, f"probe exited {completed.returncode}")
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                metrics = validate_probe_payload(
                    payload,
                    blocks=args.blocks,
                    expected_srcversion=loaded_srcversion,
                )
                validate_route_delta(enabled, delta)
                wait_for_zero_mib(args.device, args.zero_mib_timeout)

                arm = {
                    "cycle": cycle,
                    "position": position,
                    "trace_enabled": enabled,
                    "artifact": str(artifact.relative_to(args.output_dir)),
                    "route_counter_delta": delta,
                    "metrics": metrics,
                }
                arms.append(arm)
                artifact.with_suffix(".route_delta.json").write_text(
                    json.dumps(arm, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

        summary["comparison"] = comparison(arms)
        summary["status"] = "pass"
    except Exception as error:  # noqa: BLE001
        failure = f"{type(error).__name__}: {error}"
        summary["error"] = failure
    finally:
        try:
            set_route_gate(args.route_parameter, original_gate, dry_run=False)
            summary["restored_route_gate"] = read_int(args.route_parameter)
        except Exception as restore_error:  # noqa: BLE001
            summary["restore_error"] = f"{type(restore_error).__name__}: {restore_error}"
            if failure is None:
                failure = summary["restore_error"]

        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if summary.get("status") == "pass" and failure is None:
            write_report(summary, args.output_dir / "REPORT.md")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if failure is not None:
        print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
