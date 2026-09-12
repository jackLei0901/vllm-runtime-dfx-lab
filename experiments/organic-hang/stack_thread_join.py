"""Join Flight Recorder issuing threads to py-spy threads without labels or time."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_task_names(pid: int, proc_root: Path = Path("/proc")) -> dict[int, str]:
    result = {}
    for task in (proc_root / str(pid) / "task").iterdir():
        if task.name.isdigit():
            result[int(task.name)] = (task / "comm").read_text(encoding="utf-8").strip()
    return result


def match_issuing_thread(
    *,
    flight_entry: dict[str, Any],
    py_spy_threads: list[dict[str, Any]],
    task_names: dict[int, str],
) -> dict[str, Any]:
    """Match by OS TID; the kernel task name is retained only as corroboration."""
    thread_id = flight_entry.get("thread_id")
    if thread_id is None:
        return {"outcome": "flight_thread_id_unavailable", "thread": None}
    thread_id = int(thread_id)
    candidates = [
        thread
        for thread in py_spy_threads
        if int(thread.get("os_thread_id", -1)) == thread_id
    ]
    if not candidates:
        return {
            "outcome": "issuing_thread_not_sampled",
            "thread": None,
            "os_thread_id": thread_id,
            "task_name": task_names.get(thread_id),
        }
    if len(candidates) != 1:
        raise ValueError(f"multiple py-spy threads have OS TID {thread_id}")
    return {
        "outcome": "matched",
        "thread": candidates[0],
        "os_thread_id": thread_id,
        "task_name": task_names.get(thread_id),
    }


def first_project_frame(
    frames: list[dict[str, Any]], *, prepared_target: Path
) -> dict[str, Any] | None:
    """Select a frame by exact source provenance, never by a known function name."""
    target = prepared_target.resolve()
    for frame in frames:
        filename = frame.get("filename")
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path == target:
            return {
                "function": str(frame.get("name", "unknown"))[:128],
                "file": target.name,
                "line": int(frame.get("line", 0)),
            }
    return None
