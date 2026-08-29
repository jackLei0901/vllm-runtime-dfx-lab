from __future__ import annotations

import json
import os
import signal
from pathlib import Path

from dfxlab.schema import utc_now


def inject_signal(
    pid: int,
    signal_name: str,
    event_log: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    normalized = signal_name.upper()
    if not normalized.startswith("SIG"):
        normalized = f"SIG{normalized}"
    signum = getattr(signal, normalized, None)
    if signum is None:
        raise ValueError(f"unsupported signal: {signal_name}")
    event: dict[str, object] = {
        "timestamp": utc_now(),
        "kind": "signal_injection",
        "target_pid": pid,
        "signal": normalized,
        "dry_run": dry_run,
    }
    if not dry_run:
        os.kill(pid, signum)
    if event_log is not None:
        event_log.parent.mkdir(parents=True, exist_ok=True)
        with event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event

