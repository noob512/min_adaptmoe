#!/usr/bin/env python3
"""Static invariants for the production-off UVM route trace gate."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UVM = ROOT / "kernel-module/nvidia-module/kernel-open/nvidia-uvm"


def section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    proc_h = (UVM / "uvm_procfs.h").read_text()
    proc_c = (UVM / "uvm_procfs.c").read_text()
    api_h = (UVM / "uvm_api.h").read_text()
    uvm_c = (UVM / "uvm.c").read_text()

    require(
        "unsigned int uvm_phase_f_route_log_enable = 0;" in proc_c,
        "route trace must default to disabled",
    )
    require(
        "module_param(uvm_phase_f_route_log_enable, uint, "
        "S_IRUGO | S_IWUSR);" in proc_c,
        "route trace module parameter must be dynamically writable by root",
    )

    gate = section(
        proc_h,
        "#define uvm_phase_f_route_log(fmt, ...)",
        "#define UVM_SEQ_OR_DBG_PRINT",
    )
    require(
        gate.index("uvm_phase_f_route_log_is_enabled()")
        < gate.index("uvm_phase_f_route_log_emit(fmt, ##__VA_ARGS__)"),
        "header gate must dominate emit so disabled format arguments are not evaluated",
    )

    emit = section(
        proc_c,
        "void uvm_phase_f_route_log_emit(const char *fmt, ...)",
        "void uvm_phase_f_route_log_always(const char *fmt, ...)",
    )
    require(
        emit.index("if (unlikely(!uvm_phase_f_route_log_is_enabled()))")
        < emit.index("va_start(args, fmt)"),
        "emit race gate must precede vararg setup",
    )
    require(
        emit.index("if (unlikely(!uvm_phase_f_route_log_is_enabled()))")
        < emit.index("uvm_phase_f_route_vlog(fmt, args)"),
        "emit race gate must precede the timestamp/format/lock helper",
    )
    for forbidden in ("ktime_get_real_ts64", "vscnprintf", "mutex_lock"):
        require(forbidden not in emit, f"disabled emit path must not contain {forbidden}")

    vlog = section(
        proc_c,
        "static void uvm_phase_f_route_vlog(const char *fmt, va_list args)",
        "void uvm_phase_f_route_log_emit(const char *fmt, ...)",
    )
    ordered = [
        "g_phase_f_route_timestamp_calls",
        "ktime_get_real_ts64",
        "g_phase_f_route_format_calls",
        "vscnprintf",
        "g_phase_f_route_mutex_acquires",
        "mutex_lock",
    ]
    require(
        [vlog.index(token) for token in ordered]
        == sorted(vlog.index(token) for token in ordered),
        "runtime counters must bracket timestamp, format, and mutex work",
    )

    route_alloc_print = section(
        api_h,
        "#define UVM_ROUTE_ALLOC_PRINT",
        "#define UVM_ROUTE_ALLOC_ERROR",
    )
    require(
        "UVM_INFO_PRINT" not in route_alloc_print,
        "normal allocated-ioctl route events must not unconditionally hit printk",
    )
    require(
        api_h.count("UVM_ROUTE_ALLOC_ERROR(\"route_alloc ") == 3,
        "allocation and both copy_user failures must remain on UVM_ERR_PRINT",
    )

    for event in (
        "phase_f_policy_batch expert_fast_batch_fallback",
        "phase_f_policy_batch_regions expert_fast_region_batch_fallback",
    ):
        position = uvm_c.index(event)
        prefix = uvm_c[max(0, position - 80) : position]
        require(
            "uvm_phase_f_route_log_always(" in prefix,
            f"fallback event is not on the always-on low-frequency path: {event}",
        )

    require(
        'NV_CREATE_PROC_FILE("phase_f_route_log_stats"' in proc_c,
        "runtime A/B counters must be exposed through procfs",
    )

    print("route trace static invariants: PASS")


if __name__ == "__main__":
    main()
