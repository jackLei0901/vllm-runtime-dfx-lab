"""Emit one genuine four-rank NCCL Flight Recorder artifact per rank.

This is an instance smoke test, not a formal organic-hang trial. Raw pickle
files are transient and must be removed after the input contract is checked.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
from pathlib import Path

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _dump_nccl_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--skip-collective-rank", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    args.state_dir.mkdir(parents=True, exist_ok=True)
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=args.timeout_seconds)
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    value = torch.tensor([rank + 1.0], device="cuda")
    if rank == args.skip_collective_rank:
        import time

        time.sleep(args.timeout_seconds * 3)
        return 0
    dist.all_reduce(value)
    output = torch.empty(1, device="cuda")
    input_value = torch.arange(dist.get_world_size(), device="cuda").float()
    dist.reduce_scatter_tensor(output, input_value)
    torch.cuda.synchronize()

    raw = _dump_nccl_trace(True, True, False)
    decoded = pickle.loads(raw)
    raw_path = args.state_dir / f"rank-{rank}.pickle"
    raw_path.write_bytes(raw)

    entries = decoded.get("entries", [])
    summary = {
        "rank": rank,
        "top_level_keys": sorted(decoded),
        "entry_count": len(entries),
        "entry_keys": sorted({key for entry in entries for key in entry}),
        "pg_config_type": type(decoded.get("pg_config")).__name__,
        "pg_config": decoded.get("pg_config"),
    }
    (args.state_dir / f"rank-{rank}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
