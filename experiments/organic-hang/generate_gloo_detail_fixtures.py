"""Generate genuine ProcessGroupWrapper DETAIL fixtures without a GPU.

Run this in an environment with the affected PyTorch build. The output is
sanitized to the mismatch/barrier messages before it is retained.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

KEEP = re.compile(
    r"(?:Detected mismatch between collectives on ranks|"
    r"ProcessGroupWrapper: Monitored Barrier encountered error|"
    r"Ranks?\s+\d+(?:,\s*\d+)*\s+failed to pass monitoredBarrier)[^\n]*",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, mode: str) -> None:
    import datetime
    import time

    import torch
    import torch.distributed as dist

    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=5),
    )
    if mode == "long":
        size = 960 if rank == 0 else 1024
        dist.all_reduce(torch.ones(size))
    elif mode == "degraded":
        tensor = torch.ones(4) if rank == 0 else torch.ones(2, 2)
        dist.all_reduce(tensor)
    elif rank == 0:
        dist.all_reduce(torch.ones(4))
    else:
        time.sleep(8)


def _run_child(mode: str) -> None:
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=(2, mode), nprocs=2, join=True)


def _sanitize(stderr: str) -> str:
    match = KEEP.search(stderr)
    if match is None:
        raise RuntimeError("DETAIL run did not emit a supported oracle message")
    message = match.group(0)
    message = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s:]+", "<path>", message)
    return message.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child-mode", choices=["long", "degraded", "barrier"])
    args = parser.parse_args()
    if args.child_mode:
        _run_child(args.child_mode)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("long", "degraded", "barrier"):
        env = os.environ.copy()
        env.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(_free_port()),
                "TORCH_CPP_LOG_LEVEL": "ERROR",
            }
        )
        with tempfile.TemporaryDirectory(prefix="organic-gloo-") as directory:
            env["TMPDIR"] = directory
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--output-dir",
                    str(args.output_dir),
                    "--child-mode",
                    mode,
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        (args.output_dir / f"detail-{mode}.txt").write_text(
            _sanitize(result.stderr), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
