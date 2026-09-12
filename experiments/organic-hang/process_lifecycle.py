"""Bounded, PID-reuse-safe lifecycle helpers for the four-rank campaign."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

TERM_SIGNAL = getattr(signal, "SIGTERM", 15)
KILL_SIGNAL = getattr(signal, "SIGKILL", 9)


@dataclass(frozen=True)
class ProcessIdentity:
    """Linux process identity stable across a PID reuse."""

    pid: int
    start_time_ticks: int


def read_start_time(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """Return `/proc/<pid>/stat` field 22, or ``None`` after process exit."""
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    # The comm field is parenthesized and may contain spaces or parentheses.
    closing = stat.rfind(")")
    if closing < 0:
        raise ValueError(f"malformed stat record for pid {pid}")
    fields_from_state = stat[closing + 1 :].split()
    if len(fields_from_state) <= 19:
        raise ValueError(f"stat record for pid {pid} has no start time")
    return int(fields_from_state[19])


def identify(pid: int, proc_root: Path = Path("/proc")) -> ProcessIdentity:
    start_time = read_start_time(pid, proc_root)
    if start_time is None:
        raise ProcessLookupError(pid)
    return ProcessIdentity(pid=pid, start_time_ticks=start_time)


def identity_is_live(
    identity: ProcessIdentity, proc_root: Path = Path("/proc")
) -> bool:
    return read_start_time(identity.pid, proc_root) == identity.start_time_ticks


def wait_for_rank_identities(
    state_dir: Path,
    world_size: int,
    deadline_monotonic: float,
    parent_poll: Callable[[], int | None],
    proc_root: Path = Path("/proc"),
) -> list[ProcessIdentity]:
    """Wait for atomic rank PID files and bind each PID to its start time."""
    expected = [state_dir / f"rank-{rank}.pid" for rank in range(world_size)]
    while time.monotonic() < deadline_monotonic:
        if parent_poll() is not None:
            raise RuntimeError("launcher exited before all rank identities appeared")
        if all(path.exists() for path in expected):
            identities = [
                identify(int(path.read_text(encoding="utf-8").strip()), proc_root)
                for path in expected
            ]
            if len({item.pid for item in identities}) != world_size:
                raise RuntimeError("rank PID files are not unique")
            return identities
        time.sleep(0.05)
    raise TimeoutError("rank identity deadline expired")


def wait_until_exit_or_deadline(
    parent_poll: Callable[[], int | None],
    deadline_monotonic: float,
    poll_interval_seconds: float = 0.05,
) -> int | None:
    """Return the exit code, or ``None`` when the wall-clock deadline expires."""
    while time.monotonic() < deadline_monotonic:
        return_code = parent_poll()
        if return_code is not None:
            return return_code
        time.sleep(poll_interval_seconds)
    return None


def _signal_if_same(
    identity: ProcessIdentity,
    requested_signal: int,
    proc_root: Path,
) -> bool:
    if not identity_is_live(identity, proc_root):
        return False
    os.kill(identity.pid, requested_signal)
    return True


def cleanup_process_group(
    parent: subprocess.Popen[bytes] | subprocess.Popen[str],
    parent_identity: ProcessIdentity,
    rank_identities: list[ProcessIdentity],
    grace_seconds: float = 5.0,
    proc_root: Path = Path("/proc"),
) -> dict[str, object]:
    """Terminate only the recorded processes, then verify that none remain."""
    term_rank_count = sum(
        _signal_if_same(item, TERM_SIGNAL, proc_root) for item in rank_identities
    )
    term_group_sent = False
    if identity_is_live(parent_identity, proc_root):
        os.killpg(parent_identity.pid, TERM_SIGNAL)
        term_group_sent = True
    try:
        parent.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass

    kill_rank_count = sum(
        _signal_if_same(item, KILL_SIGNAL, proc_root) for item in rank_identities
    )
    kill_group_sent = False
    if identity_is_live(parent_identity, proc_root):
        os.killpg(parent_identity.pid, KILL_SIGNAL)
        kill_group_sent = True
    try:
        parent.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass

    deadline = time.monotonic() + grace_seconds
    tracked = [parent_identity, *rank_identities]
    while time.monotonic() < deadline and any(
        identity_is_live(item, proc_root) for item in tracked
    ):
        time.sleep(0.05)
    remaining = [item.pid for item in tracked if identity_is_live(item, proc_root)]
    return {
        "pid_start_times_verified": True,
        "term_rank_count": term_rank_count,
        "term_group_sent": term_group_sent,
        "kill_rank_count": kill_rank_count,
        "kill_group_sent": kill_group_sent,
        "remaining_tracked_pids": remaining,
        "no_orphans": not remaining,
    }


def start_campaign_process(
    command: list[str],
    environment: dict[str, str],
    stdout: int | object = subprocess.PIPE,
    stderr: int | object = subprocess.PIPE,
) -> tuple[subprocess.Popen[bytes], ProcessIdentity]:
    """Start the campaign as a new POSIX session and record the leader identity."""
    if os.name != "posix":
        raise OSError("the four-rank campaign launcher requires Linux /proc")
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    return process, identify(process.pid)
