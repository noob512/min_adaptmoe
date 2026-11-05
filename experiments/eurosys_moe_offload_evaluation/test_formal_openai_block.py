from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module():
    path = ROOT / "formal_openai_block.py"
    spec = importlib.util.spec_from_file_location("formal_openai_block", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _request(token_count: int) -> dict[str, object]:
    return {
        "success": True,
        "http_status": 200,
        "completion_tokens": token_count,
        "output_token_ids": list(range(token_count)),
        "output_token_ids_sha256": "hash",
        "saw_done": True,
        "grouped_token_events": 0,
        "fallback_token_events": 0,
    }


def test_client_checks_accepts_exact_successful_phases() -> None:
    module = load_module()
    client = {
        "status": "ok",
        "config": {"output_tokens": 4},
        "phases": [
            {
                "concurrency": 1,
                "successful_requests": 1,
                "failed_requests": 0,
                "requests": [_request(4)],
                "ttft_ms": {"mean": 1.0},
                "itl_ms": {"mean": 2.0},
                "e2e_ms": {"mean": 7.0},
            },
            {
                "concurrency": 2,
                "successful_requests": 2,
                "failed_requests": 0,
                "requests": [_request(4), _request(4)],
                "ttft_ms": {"mean": 3.0},
                "itl_ms": {"mean": 4.0},
                "e2e_ms": {"mean": 15.0},
            },
        ],
    }

    checks, failures = module.client_checks(client)

    assert failures == []
    assert [phase["ok"] for phase in checks["phases"]] == [True, True]


def test_client_checks_rejects_short_output() -> None:
    module = load_module()
    request = _request(3)
    client = {
        "status": "ok",
        "config": {"output_tokens": 4},
        "phases": [
            {
                "concurrency": 1,
                "successful_requests": 1,
                "failed_requests": 0,
                "requests": [request],
            }
        ],
    }

    _, failures = module.client_checks(client)

    assert any("protocol checks" in failure for failure in failures)


def test_workload_config_requires_frozen_protocol_and_files(
    tmp_path: Path,
) -> None:
    module = load_module()
    evaluation = tmp_path / "evaluation.json"
    warmup = tmp_path / "warmup.json"
    evaluation.write_text(json.dumps({}), encoding="utf-8")
    warmup.write_text(json.dumps({}), encoding="utf-8")
    value = {
        "schema_version": 1,
        "protocol_id": "protocol-v1",
        "workloads": [
            {
                "name": "short_dialogue",
                "evaluation_fixture": str(evaluation),
                "warmup_fixture": str(warmup),
                "concurrencies": [1, 4],
            }
        ],
    }

    workloads = module.validate_workload_config(value, "protocol-v1")

    assert workloads[0]["concurrencies"] == [1, 4]


def test_log_scan_tolerates_only_clean_ktransformers_shutdown_traceback() -> None:
    module = load_module()
    log = """Process Process-1:
Traceback (most recent call last):
  File \"scheduler.py\", line 1, in event_loop
KeyboardInterrupt
INFO Finished server process [123]
"""

    fatal, ignored = module.scan_fatal_log("ktransformers", log)

    assert fatal == []
    assert ignored == ["ktransformers_shutdown_keyboard_interrupt_traceback"]


def test_log_scan_rejects_real_ktransformers_traceback() -> None:
    module = load_module()
    log = """Traceback (most recent call last):
  File \"jit.py\", line 1, in build
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
"""

    fatal, ignored = module.scan_fatal_log("ktransformers", log)

    assert re_escape(module.TRACEBACK_MARKER) in fatal
    assert ignored == []


def re_escape(value: str) -> str:
    import re

    return re.escape(value)
