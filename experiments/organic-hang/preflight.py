"""Fail-closed preflight for the four-GPU organic hang campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from fetch_and_prepare_reproducer import PREPARED_SHA256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-target", type=Path, required=True)
    parser.add_argument("--expected-prepared-sha256", default=PREPARED_SHA256)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-cuda-version", required=True)
    parser.add_argument("--py-spy", default="py-spy")
    args = parser.parse_args()

    import torch

    prepared = args.prepared_target.read_bytes()
    observed_hash = hashlib.sha256(prepared).hexdigest()
    if observed_hash != args.expected_prepared_sha256:
        raise SystemExit(
            "prepared target mismatch: expected "
            f"{args.expected_prepared_sha256}, "
            f"observed {observed_hash}"
        )
    if torch.__version__ != args.expected_torch_version:
        raise SystemExit(
            f"torch version mismatch: expected {args.expected_torch_version}, "
            f"observed {torch.__version__}"
        )
    if torch.version.cuda != args.expected_cuda_version:
        raise SystemExit(
            f"CUDA runtime mismatch: expected {args.expected_cuda_version}, "
            f"observed {torch.version.cuda}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise SystemExit(
            "exactly four visible CUDA devices are required; set "
            "CUDA_VISIBLE_DEVICES before preflight"
        )
    gpu_names = [torch.cuda.get_device_name(index) for index in range(4)]
    if len(set(gpu_names)) != 1:
        raise SystemExit(f"four identical GPU models are required: {gpu_names}")

    py_spy = subprocess.run(
        [args.py_spy, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if py_spy.returncode != 0:
        raise SystemExit("py-spy is unavailable")

    driver_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if driver_query.returncode != 0:
        raise SystemExit("NVIDIA driver version is unavailable")
    driver_versions = {
        line.strip() for line in driver_query.stdout.splitlines() if line.strip()
    }
    if len(driver_versions) != 1:
        raise SystemExit(f"GPU driver versions disagree: {sorted(driver_versions)}")

    ptrace_scope = None
    ptrace_path = Path("/proc/sys/kernel/yama/ptrace_scope")
    if ptrace_path.exists():
        ptrace_scope = ptrace_path.read_text(encoding="utf-8").strip()

    result = {
        "status": "PASS",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "gpu_model": gpu_names[0],
        "driver": driver_versions.pop(),
        "visible_gpu_count": 4,
        "prepared_sha256": observed_hash,
        "py_spy": (py_spy.stdout or py_spy.stderr).strip(),
        "ptrace_scope": ptrace_scope,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
