#!/usr/bin/env python3
"""Run one immutable formal server process block with the common SSE client."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FATAL_PATTERNS = (
    re.compile(r"CUDA error", re.IGNORECASE),
    re.compile(r"index out of bounds", re.IGNORECASE),
    re.compile(r"segmentation fault", re.IGNORECASE),
    re.compile(r"\bXid\b", re.IGNORECASE),
    re.compile(r"PDE.*fault", re.IGNORECASE),
)
TRACEBACK_MARKER = "Traceback (most recent call last):"
PYTHON_ERROR_LINE = re.compile(
    r"(?m)^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)(?::.*)?$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_text(*command: str, check: bool = False) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.returncode, completed.stdout


def gpu_state() -> dict[str, Any]:
    query = (
        "index,uuid,name,memory.used,memory.free,temperature.gpu,"
        "utilization.gpu,power.draw,clocks.current.sm,clocks.current.memory,"
        "pcie.link.gen.current,pcie.link.width.current"
    )
    rc, gpu_output = run_text(
        "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"
    )
    app_rc, app_output = run_text(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    return {
        "created_at": utc_now(),
        "gpu_query_returncode": rc,
        "gpu_query": query,
        "gpu_rows": [line for line in gpu_output.splitlines() if line.strip()],
        "compute_query_returncode": app_rc,
        "compute_process_rows": [
            line for line in app_output.splitlines() if line.strip()
        ],
    }


def gpu_is_clean() -> bool:
    state = gpu_state()
    if state["gpu_query_returncode"] != 0 or state["compute_query_returncode"] != 0:
        return False
    if state["compute_process_rows"]:
        return False
    rows = state["gpu_rows"]
    if len(rows) != 1:
        return False
    fields = [value.strip() for value in rows[0].split(",")]
    try:
        memory_used_mib = int(fields[3])
    except (IndexError, ValueError):
        return False
    return memory_used_mib == 0


def wait_gpu_clean(timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if gpu_is_clean():
            return True
        time.sleep(1.0)
    return gpu_is_clean()


def health_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def fetch_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} from {url}")
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object JSON from {url}")
    return value


def wait_ready(
    *, process: subprocess.Popen[Any], health_url: str, timeout_seconds: float
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if health_ok(health_url):
            return True
        time.sleep(1.0)
    return health_ok(health_url)


def terminate_process_group(
    process: subprocess.Popen[Any], *, grace_seconds: float = 30.0
) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return int(process.wait())
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            return int(returncode)
        time.sleep(0.25)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(process.wait())
    try:
        return int(process.wait(timeout=10.0))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait())


def start_nvml_sampler(output: Path, error: Path) -> tuple[subprocess.Popen[Any], Any, Any]:
    fields = (
        "timestamp,temperature.gpu,utilization.gpu,utilization.memory,"
        "memory.used,memory.free,power.draw,clocks.current.sm,"
        "clocks.current.memory,pstate"
    )
    output_handle = output.open("xb")
    error_handle = error.open("xb")
    try:
        process = subprocess.Popen(
            (
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ),
            stdout=output_handle,
            stderr=error_handle,
            start_new_session=True,
        )
    except BaseException:
        output_handle.close()
        error_handle.close()
        raise
    return process, output_handle, error_handle


def stop_sampler(process: subprocess.Popen[Any], *handles: Any) -> int:
    if process.poll() is None:
        process.terminate()
    try:
        returncode = int(process.wait(timeout=5.0))
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = int(process.wait())
    for handle in handles:
        handle.close()
    return returncode


def validate_workload_config(value: Any, protocol_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("workload config must use schema_version=1")
    if value.get("protocol_id") != protocol_id:
        raise ValueError("workload config protocol differs")
    workloads = value.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workload config has no workloads")
    names: set[str] = set()
    for workload in workloads:
        if not isinstance(workload, dict):
            raise ValueError("workload entry must be an object")
        name = workload.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", name):
            raise ValueError(f"invalid workload name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate workload: {name}")
        names.add(name)
        for key in ("evaluation_fixture", "warmup_fixture"):
            path = Path(str(workload.get(key, "")))
            if not path.is_file():
                raise ValueError(f"missing {key}: {path}")
        concurrencies = workload.get("concurrencies")
        if (
            not isinstance(concurrencies, list)
            or not concurrencies
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in concurrencies
            )
        ):
            raise ValueError(f"invalid concurrencies for {name}")
        measurement_requests = workload.get("measurement_requests", max(concurrencies))
        if (
            isinstance(measurement_requests, bool)
            or not isinstance(measurement_requests, int)
            or measurement_requests <= 0
            or any(measurement_requests % value for value in concurrencies)
        ):
            raise ValueError(f"invalid measurement_requests for {name}")
        minimum_interval = workload.get("minimum_request_start_interval_seconds", 0.0)
        if (
            isinstance(minimum_interval, bool)
            or not isinstance(minimum_interval, (int, float))
            or minimum_interval < 0
            or (minimum_interval > 0 and max(concurrencies) != 1)
        ):
            raise ValueError(f"invalid request-start interval for {name}")
    return workloads


def client_checks(client: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    phases = client.get("phases")
    config = client.get("config", {})
    expected_output_tokens = config.get("output_tokens")
    configured_requests = config.get("measurement_requests")
    minimum_interval = config.get("minimum_request_start_interval_seconds", 0.0)
    if client.get("status") != "ok":
        failures.append(f"client status={client.get('status')!r}")
    if not isinstance(phases, list) or not phases:
        failures.append("client has no measurement phases")
        phases = []
    phase_checks: list[dict[str, Any]] = []
    for phase in phases:
        requests = phase.get("requests") if isinstance(phase, dict) else None
        concurrency = phase.get("concurrency") if isinstance(phase, dict) else None
        expected_requests = (
            configured_requests
            if isinstance(configured_requests, int)
            else concurrency
        )
        observed_request_count = (
            phase.get("request_count", len(requests))
            if isinstance(phase, dict) and isinstance(requests, list)
            else None
        )
        rate_compliant = (
            phase.get("request_rate_limit_compliant", minimum_interval == 0)
            if isinstance(phase, dict)
            else False
        )
        phase_ok = (
            isinstance(requests, list)
            and isinstance(concurrency, int)
            and len(requests) == expected_requests
            and observed_request_count == expected_requests
            and phase.get("successful_requests") == expected_requests
            and phase.get("failed_requests") == 0
            and rate_compliant is True
        )
        hashes: list[str] = []
        if isinstance(requests, list):
            for index, request in enumerate(requests):
                request_ok = (
                    isinstance(request, dict)
                    and request.get("success") is True
                    and request.get("http_status") == 200
                    and request.get("completion_tokens") == expected_output_tokens
                    and isinstance(request.get("output_token_ids"), list)
                    and len(request["output_token_ids"]) == expected_output_tokens
                    and request.get("saw_done") is True
                    and request.get("grouped_token_events") == 0
                    and request.get("fallback_token_events") == 0
                )
                if not request_ok:
                    failures.append(
                        f"concurrency={concurrency} request={index} failed protocol checks"
                    )
                    phase_ok = False
                if isinstance(request, dict) and isinstance(
                    request.get("output_token_ids_sha256"), str
                ):
                    hashes.append(request["output_token_ids_sha256"])
        if not phase_ok:
            failures.append(f"concurrency={concurrency} phase counts differ")
        phase_checks.append(
            {
                "concurrency": concurrency,
                "ok": phase_ok,
                "request_count": len(requests) if isinstance(requests, list) else None,
                "minimum_request_start_interval_seconds": minimum_interval,
                "observed_minimum_request_start_interval_seconds": (
                    phase.get("observed_minimum_request_start_interval_seconds")
                    if isinstance(phase, dict)
                    else None
                ),
                "request_output_hashes": hashes,
                "mean_ttft_ms": phase.get("ttft_ms", {}).get("mean")
                if isinstance(phase, dict)
                else None,
                "mean_tpot_ms": phase.get("itl_ms", {}).get("mean")
                if isinstance(phase, dict)
                else None,
                "mean_e2e_ms": phase.get("e2e_ms", {}).get("mean")
                if isinstance(phase, dict)
                else None,
            }
        )
    return {"status": client.get("status"), "phases": phase_checks}, failures


def scan_fatal_log(system: str, server_log: str) -> tuple[list[str], list[str]]:
    """Return fatal and explicitly tolerated server-log signatures.

    SGLang stops its scheduler and detokenizer children with SIGINT.  Those
    children print traceback-shaped ``KeyboardInterrupt`` records even after
    every request has completed and the parent exits cleanly.  Tolerate only
    that narrow shutdown form; a traceback ending in any Error/Exception
    class remains fatal.
    """
    fatal = sorted(
        {
            pattern.pattern
            for pattern in FATAL_PATTERNS
            if pattern.search(server_log)
        }
    )
    ignored: list[str] = []
    if TRACEBACK_MARKER in server_log:
        python_errors = sorted(set(PYTHON_ERROR_LINE.findall(server_log)))
        expected_kt_shutdown = (
            system == "ktransformers"
            and not python_errors
            and "\nKeyboardInterrupt\n" in server_log
            and "Finished server process" in server_log
        )
        if expected_kt_shutdown:
            ignored.append("ktransformers_shutdown_keyboard_interrupt_traceback")
        else:
            fatal.append(re.escape(TRACEBACK_MARKER))
    return sorted(set(fatal)), ignored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument(
        "--system", required=True, choices=("ours", "ktransformers", "moe_infinity")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workload-config", required=True, type=Path)
    parser.add_argument(
        "--workload-filter",
        help=(
            "Optional comma-separated exact workload names. The immutable "
            "source config remains recorded, while only the selected cells "
            "run in this server process block."
        ),
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        help=(
            "Replace each workload fixture path with a file of the same "
            "name under this model-specific frozen fixture directory"
        ),
    )
    parser.add_argument("--server-env", required=True, type=Path)
    parser.add_argument(
        "--server-env-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--streaming-client", required=True, type=Path)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--gpu-clean-timeout", type=float, default=120.0)
    parser.add_argument(
        "--stop-on-first-workload-failure",
        action="store_true",
        help="terminate the server block after the first failed workload",
    )
    parser.add_argument(
        "--stop-after-ready",
        action="store_true",
        help="diagnostic mode: terminate immediately after server readiness",
    )
    parser.add_argument("server_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.server_command and args.server_command[0] == "--":
        args.server_command = args.server_command[1:]
    if not args.server_command:
        parser.error("server command is required after --")
    return args


def main() -> int:
    args = parse_args()
    if args.case_root.exists():
        raise SystemExit(f"case root already exists: {args.case_root}")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise SystemExit(f"Python is not executable: {args.python}")
    if not args.streaming_client.is_file():
        raise SystemExit(f"missing streaming client: {args.streaming_client}")
    workload_config = json.loads(args.workload_config.read_text(encoding="utf-8"))
    if args.fixture_root is not None:
        fixture_root = args.fixture_root.resolve()
        for workload in workload_config.get("workloads", []):
            for key in ("evaluation_fixture", "warmup_fixture"):
                original = Path(str(workload.get(key, "")))
                workload[key] = str(fixture_root / original.name)
    workloads = validate_workload_config(workload_config, args.protocol_id)
    if args.workload_filter:
        selected_names = {
            item.strip() for item in args.workload_filter.split(",") if item.strip()
        }
        known_names = {str(workload["name"]) for workload in workloads}
        unknown_names = sorted(selected_names - known_names)
        if unknown_names:
            raise SystemExit(f"unknown workload filter names: {unknown_names!r}")
        workloads = [
            workload for workload in workloads if workload["name"] in selected_names
        ]
        if not workloads:
            raise SystemExit("workload filter selected no workloads")
    if args.stop_after_ready:
        workloads = []
    strict_local_rate_limit = all(
        max(workload["concurrencies"]) == 1
        and float(
            workload.get("minimum_request_start_interval_seconds", 0.0)
        ) > 6.0
        for workload in workloads
    )
    server_env_value = json.loads(args.server_env.read_text(encoding="utf-8"))
    if not isinstance(server_env_value, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in server_env_value.items()
    ):
        raise SystemExit("server env must be a JSON object of strings")
    for item in args.server_env_override:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"invalid --server-env-override: {item!r}")
        server_env_value[key] = value
    base_url = args.base_url.rstrip("/")
    health_url = base_url + "/health"
    if health_ok(health_url):
        raise SystemExit(f"health endpoint is already serving: {health_url}")
    if not wait_gpu_clean(args.gpu_clean_timeout):
        raise SystemExit("GPU is not clean before formal case")

    args.case_root.mkdir(parents=True)
    atomic_json(args.case_root / "gpu.before.json", gpu_state())
    (args.case_root / "command.txt").write_text(
        shlex.join(args.server_command) + "\n", encoding="utf-8"
    )
    atomic_json(
        args.case_root / "env.json",
        {
            "schema_version": 1,
            "overrides": server_env_value,
            "request_rate_policy": (
                "all inference POST starts >6 seconds apart (<=10/minute), "
                "including localhost benchmark traffic"
                if strict_local_rate_limit
                else "external requests <= 10/minute; localhost benchmark "
                "traffic excluded"
            ),
        },
    )
    atomic_json(
        args.case_root / "inputs.json",
        {
            "schema_version": 1,
            "protocol_id": args.protocol_id,
            "workload_config": {
                "path": str(args.workload_config.resolve()),
                "sha256": sha256_file(args.workload_config),
                "selected_workloads": [
                    str(workload["name"]) for workload in workloads
                ],
            },
            "server_env": {
                "path": str(args.server_env.resolve()),
                "sha256": sha256_file(args.server_env),
            },
            "streaming_client": {
                "path": str(args.streaming_client.resolve()),
                "sha256": sha256_file(args.streaming_client),
            },
        },
    )

    server_log_handle = (args.case_root / "server.log").open("xb")
    sampler, sampler_output, sampler_error = start_nvml_sampler(
        args.case_root / "nvml.csv", args.case_root / "nvml.err"
    )
    environment = os.environ.copy()
    environment.update(server_env_value)
    server = subprocess.Popen(
        args.server_command,
        env=environment,
        stdout=server_log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (args.case_root / "server.pid").write_text(f"{server.pid}\n", encoding="utf-8")

    started_at = utc_now()
    ready = False
    server_returncode: int | None = None
    sampler_returncode: int | None = None
    client_records: list[dict[str, Any]] = []
    execution_failures: list[str] = []
    try:
        ready = wait_ready(
            process=server,
            health_url=health_url,
            timeout_seconds=args.ready_timeout,
        )
        if not ready:
            execution_failures.append("server did not reach healthy state")
        else:
            atomic_json(args.case_root / "gpu.ready.json", gpu_state())
            for workload in workloads:
                if server.poll() is not None:
                    execution_failures.append(
                        f"server exited before workload {workload['name']}"
                    )
                    break
                workload_root = args.case_root / workload["name"]
                workload_root.mkdir()
                client_output = workload_root / "client.json"
                client_log = workload_root / "client.log"
                command = (
                    str(args.python),
                    str(args.streaming_client),
                    "--system",
                    args.system,
                    "--model",
                    args.model,
                    "--fixture",
                    str(Path(workload["evaluation_fixture"]).resolve()),
                    "--warmup-fixture",
                    str(Path(workload["warmup_fixture"]).resolve()),
                    "--output",
                    str(client_output.resolve()),
                    "--base-url",
                    base_url,
                    "--concurrencies",
                    ",".join(str(value) for value in workload["concurrencies"]),
                    "--warmup-requests",
                    str(workload.get("warmup_requests", 3)),
                    "--warmup-output-tokens",
                    str(workload.get("warmup_output_tokens", 8)),
                    "--measurement-requests",
                    str(workload.get("measurement_requests", max(workload["concurrencies"]))),
                    "--minimum-request-start-interval",
                    str(workload.get("minimum_request_start_interval_seconds", 0.0)),
                    "--request-timeout",
                    str(args.request_timeout),
                    "--barrier-timeout",
                    str(workload.get("barrier_timeout", 60.0)),
                    "--seed",
                    str(workload.get("seed", 20260823)),
                )
                (workload_root / "command.txt").write_text(
                    shlex.join(command) + "\n", encoding="utf-8"
                )
                client_environment = os.environ.copy()
                client_environment.update(
                    {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                    }
                )
                with client_log.open("xb") as client_log_handle:
                    completed = subprocess.run(
                        command,
                        env=client_environment,
                        stdout=client_log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                (workload_root / "returncode.txt").write_text(
                    f"{completed.returncode}\n", encoding="utf-8"
                )
                record = {
                    "name": workload["name"],
                    "client": str(client_output),
                    "returncode": completed.returncode,
                    "status": None,
                }
                if client_output.is_file():
                    try:
                        client_value = json.loads(
                            client_output.read_text(encoding="utf-8")
                        )
                        record["status"] = client_value.get("status")
                        child_checks, child_failures = client_checks(client_value)
                        atomic_json(
                            workload_root / "verification.json",
                            {
                                "schema_version": 1,
                                "status": "ok" if not child_failures else "error",
                                "scope": "formal_performance_candidate",
                                "protocol_id": args.protocol_id,
                                "system": args.system,
                                "workload_case": workload["name"],
                                "checks": child_checks,
                                "failures": child_failures,
                            },
                        )
                    except (OSError, json.JSONDecodeError) as exc:
                        execution_failures.append(
                            f"client output unreadable for {workload['name']}: {exc}"
                        )
                if completed.returncode != 0 or record["status"] != "ok":
                    execution_failures.append(
                        f"client failed for {workload['name']}: "
                        f"returncode={completed.returncode}, status={record['status']!r}"
                    )
                    client_records.append(record)
                    server_unhealthy = (
                        server.poll() is not None or not health_ok(health_url)
                    )
                    if server_unhealthy:
                        execution_failures.append(
                            f"server unhealthy after workload {workload['name']}"
                        )
                    if args.stop_on_first_workload_failure or server_unhealthy:
                        break
                    continue
                client_records.append(record)
            if args.system == "moe_infinity" and server.poll() is None:
                try:
                    atomic_json(
                        args.case_root / "admin_stats.json",
                        fetch_json(base_url + "/admin/stats"),
                    )
                except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                    execution_failures.append(
                        f"MoE-Infinity admin stats unavailable: {exc}"
                    )
    except BaseException as exc:
        execution_failures.append(f"runner exception: {type(exc).__name__}: {exc}")
    finally:
        server_returncode = terminate_process_group(server)
        server_log_handle.close()
        sampler_returncode = stop_sampler(
            sampler, sampler_output, sampler_error
        )

    gpu_cleanup = wait_gpu_clean(args.gpu_clean_timeout)
    atomic_json(args.case_root / "gpu.after.json", gpu_state())
    (args.case_root / "server_returncode.txt").write_text(
        f"{server_returncode}\n", encoding="utf-8"
    )
    (args.case_root / "nvml_returncode.txt").write_text(
        f"{sampler_returncode}\n", encoding="utf-8"
    )

    server_log = (args.case_root / "server.log").read_text(
        encoding="utf-8", errors="replace"
    )
    fatal_matches, ignored_log_matches = scan_fatal_log(args.system, server_log)
    failures = list(execution_failures)
    checks: dict[str, Any] = {
        "server_ready": ready,
        "server_returncode": server_returncode,
        "nvml_sampler_returncode": sampler_returncode,
        "gpu_cleanup": gpu_cleanup,
        "fatal_log_patterns": fatal_matches,
        "ignored_log_patterns": ignored_log_matches,
        "clients": {},
    }
    if not ready:
        failures.append("server readiness check failed")
    if server_returncode not in (0, -signal.SIGINT, 130):
        failures.append(f"unexpected server returncode={server_returncode}")
    if not gpu_cleanup:
        failures.append("GPU did not clean up")
    if fatal_matches:
        failures.append(f"fatal server log patterns: {fatal_matches}")
    for record in client_records:
        client_path = Path(record["client"])
        if not client_path.is_file():
            failures.append(f"missing client output: {client_path}")
            continue
        try:
            client = json.loads(client_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"malformed client output {client_path}: {exc}")
            continue
        client_check, client_failures = client_checks(client)
        checks["clients"][record["name"]] = client_check
        failures.extend(
            f"{record['name']}: {failure}" for failure in client_failures
        )

    if args.system == "moe_infinity":
        stats_path = args.case_root / "admin_stats.json"
        stats_checks: dict[str, Any] = {"available": stats_path.is_file()}
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            expected_requests = 0
            expected_tokens = 0
            for record in client_records:
                client_path = Path(record["client"])
                if not client_path.is_file():
                    continue
                client = json.loads(client_path.read_text(encoding="utf-8"))
                config = client.get("config", {})
                warmups = int(config.get("warmup_requests", 0))
                measured = int(config.get("measurement_requests", 1)) * len(
                    config.get("concurrencies", [])
                )
                expected_requests += warmups + measured
                expected_tokens += (
                    warmups * int(config.get("warmup_output_tokens", 0))
                    + measured * int(config.get("output_tokens", 0))
                )
            stats_checks.update(
                {
                    "expected_completed_requests": expected_requests,
                    "observed_completed_requests": stats.get("completed_requests"),
                    "expected_generated_tokens": expected_tokens,
                    "observed_generated_tokens": stats.get("total_generated_tokens"),
                    "pending_requests": stats.get("pending_requests"),
                    "cancelled_requests": stats.get("cancelled_requests"),
                    "kv_cache_num_blocks": stats.get("kv_cache_num_blocks"),
                    "kv_cache_free_blocks": stats.get("kv_cache_free_blocks"),
                }
            )
            stats_ok = (
                stats.get("completed_requests") == expected_requests
                and stats.get("total_generated_tokens") == expected_tokens
                and stats.get("pending_requests") == 0
                and stats.get("cancelled_requests") == 0
                and stats.get("kv_cache_num_blocks")
                == stats.get("kv_cache_free_blocks")
            )
            stats_checks["clean"] = stats_ok
            if not stats_ok:
                failures.append("MoE-Infinity admin counters are not clean")
        else:
            stats_checks["clean"] = False
            failures.append("missing MoE-Infinity admin stats")
        checks["moe_infinity_admin_stats"] = stats_checks

    result = {
        "schema_version": 1,
        "status": "ok" if not failures else "error",
        "scope": (
            "startup_graph_capture_diagnostic"
            if args.stop_after_ready
            else "formal_performance_candidate"
        ),
        "protocol_id": args.protocol_id,
        "system": args.system,
        "started_at": started_at,
        "finished_at": utc_now(),
        "checks": checks,
        "failures": failures,
    }
    atomic_json(args.case_root / "verification.json", result)
    receipt = {
        "schema_version": 1,
        "status": "completed" if not failures else "failed",
        "verification": "verification.json",
        "verification_status": result["status"],
        "server_returncode": server_returncode,
        "gpu_cleanup_verified": gpu_cleanup,
    }
    atomic_json(
        args.case_root / ("COMPLETED.json" if not failures else "FAILED.json"),
        receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
