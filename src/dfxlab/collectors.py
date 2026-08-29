from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dfxlab.prometheus import select_metrics
from dfxlab.schema import Sample, anonymize_id, utc_now


def _http_get(url: str, timeout: float) -> tuple[int | None, str, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8"), None
    except urllib.error.HTTPError as exc:
        return exc.code, "", f"HTTPError: {exc}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, "", f"{type(exc).__name__}: {exc}"


def collect_health(base_url: str, timeout: float) -> dict[str, Any]:
    status, _, error = _http_get(f"{base_url.rstrip('/')}/health", timeout)
    return {"ok": status == 200, "status": status, "error": error}


def collect_metrics(base_url: str, timeout: float) -> tuple[dict[str, float], str | None]:
    status, body, error = _http_get(f"{base_url.rstrip('/')}/metrics", timeout)
    if status != 200:
        return {}, error or f"unexpected HTTP status {status}"
    return select_metrics(body), None


def process_snapshot(pid: int | None) -> dict[str, Any]:
    if pid is None:
        return {}
    result: dict[str, Any] = {"pid": pid, "alive": False}
    if os.name == "nt":
        result["alive"] = _windows_pid_alive(pid)
        if not result["alive"]:
            return result
    else:
        try:
            os.kill(pid, 0)
            result["alive"] = True
        except (OSError, ProcessLookupError):
            return result

    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists():
        wanted = {"VmRSS": "rss_kib", "VmSize": "vms_kib", "Threads": "threads"}
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key in wanted:
                token = value.strip().split()[0]
                result[wanted[key]] = int(token)
    return result


def _windows_pid_alive(pid: int) -> bool:
    """Check process state without psutil; os.kill(pid, 0) is unreliable on Windows."""
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def gpu_snapshot(timeout: float = 2.0) -> list[dict[str, Any]]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    fields = "index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu"
    try:
        completed = subprocess.run(
            [binary, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 6:
            continue
        gpus.append(
            {
                "index": int(values[0]),
                "uuid": anonymize_id(values[1]),
                "memory_used_mib": int(values[2]),
                "memory_total_mib": int(values[3]),
                "utilization_gpu_percent": int(values[4]),
                "temperature_c": int(values[5]),
            }
        )
    return gpus


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "gpus": gpu_snapshot(),
    }
    for module_name in ("torch", "vllm"):
        try:
            module = __import__(module_name)
            snapshot[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            snapshot[module_name] = None
    if shutil.which("git"):
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        if completed.returncode == 0:
            snapshot["git_commit"] = completed.stdout.strip()
    return snapshot


def collect_sample(base_url: str, pid: int | None, timeout: float) -> Sample:
    health = collect_health(base_url, timeout)
    metrics, metrics_error = collect_metrics(base_url, timeout)
    events: list[dict[str, Any]] = []
    if metrics_error:
        events.append({"kind": "metrics_error", "detail": metrics_error})
    return Sample(
        timestamp=utc_now(),
        monotonic_seconds=time.monotonic(),
        process=process_snapshot(pid),
        health=health,
        metrics=metrics,
        gpus=gpu_snapshot(),
        events=events,
    )


def print_environment() -> None:
    print(json.dumps(environment_snapshot(), ensure_ascii=False, indent=2))
