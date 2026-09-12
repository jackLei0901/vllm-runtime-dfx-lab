"""Fetch and deterministically prepare the pinned PyTorch #158719 reproducer."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = (
    "https://gist.githubusercontent.com/man2machine/"
    "170c4fdd44ab731441910e542d71b24e/raw/"
    "fbd907f905a5e7ab61181ad35f2ffb34508e8c83/pp_fsdp_graph_test.py"
)
SOURCE_SHA256 = "f23a41bebca5bcac51c6433ecc4a837fa3bbd1b5fd552c6701373619fafe0654"
PREPARED_SHA256 = "47fce815563b5ce8e94f93e18f1184b73fa0382ccd3934b7df02a90590c0a5a2"
FIXED_PREPARED_SHA256 = (
    "430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1"
)
NO_DIVERGENCE_PREPARED_SHA256 = (
    "bb4126cd3a74a57d1b58b11035876753871e599076728c559c3a5f3d46fa0201"
)
DEBUG_BLOCK = (
    "    if debug:  # need to enable to produce FSDP error, otherwise it will "
    "hang and not show the error\n"
    "        torch.autograd.set_detect_anomaly(True)\n"
    "        dist.set_debug_level(dist.DebugLevel.DETAIL)  # type: ignore\n"
)


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one source anchor, found {count}: {old!r}")
    return source.replace(old, new, 1)


def prepare_source(source: str) -> str:
    """Apply observability/configuration hooks without changing the fault."""
    source = replace_once(
        source, "import os\nimport abc", "import os\nimport ctypes\nimport abc"
    )
    source = replace_once(
        source,
        "ModuleT = TypeVar('ModuleT', bound=nn.Module)\n",
        """PR_SET_PTRACER = 0x59616D61


def authorize_observer_from_env() -> None:
    observer_pid = int(os.environ.get("DFX_OBSERVER_PID", "0"))
    if observer_pid <= 0:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PTRACER, observer_pid, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


ModuleT = TypeVar('ModuleT', bound=nn.Module)
""",
    )
    source = replace_once(
        source,
        "    global_rank = int(os.environ[GLOBAL_RANK_ENV_VAR])\n",
        """    global_rank = int(os.environ[GLOBAL_RANK_ENV_VAR])
    authorize_observer_from_env()
""",
    )
    source = replace_once(
        source,
        DEBUG_BLOCK,
        """    # Keep anomaly detection disabled in every arm. The oracle arm changes
    # only ProcessGroupWrapper DETAIL checking, not backward synchronization.
    torch.autograd.set_detect_anomaly(False)
    if debug:
        dist.set_debug_level(dist.DebugLevel.DETAIL)  # type: ignore
""",
    )
    source = replace_once(
        source,
        "        timeout=datetime.timedelta(60)\n",
        """        timeout=datetime.timedelta(
            seconds=int(os.environ.get("DFX_ORGANIC_TIMEOUT_S", "60"))
        )
""",
    )
    source = replace_once(
        source,
        "    world_device_mesh = parallel_dims.build_mesh(device_type='cuda')\n",
        """    state_dir = os.environ.get("DFX_STATE_DIR")
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, f"rank-{global_rank}.pid")
        temporary_state_path = state_path + ".tmp"
        with open(temporary_state_path, "w", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\\n")
        os.replace(temporary_state_path, state_path)

    world_device_mesh = parallel_dims.build_mesh(device_type='cuda')
    # DeviceMesh creates child process groups after init_process_group. In
    # PyTorch 2.11 those groups otherwise retain the 10-minute NCCL default,
    # which would bypass the experiment's explicit bounded timeout.
    group_timeout = datetime.timedelta(
        seconds=int(os.environ.get("DFX_ORGANIC_TIMEOUT_S", "60"))
    )
    for process_group in world_device_mesh.get_all_groups():
        dist.distributed_c10d._set_pg_timeout(group_timeout, process_group)
""",
    )
    source = replace_once(
        source,
        "    os.environ[MASTER_PORT_ENV_VAR] = '8000'\n",
        """    os.environ[MASTER_PORT_ENV_VAR] = os.environ.get(
        "DFX_ORGANIC_MASTER_PORT", "29500"
    )
""",
    )
    source = replace_once(
        source,
        "        total_training_steps=200,\n",
        """        total_training_steps=int(
            os.environ.get("DFX_ORGANIC_TRAINING_STEPS", "200")
        ),
""",
    )
    source = replace_once(
        source,
        "        debug=True\n",
        """        debug=os.environ.get("DFX_ORGANIC_DEBUG_DETAIL", "0") == "1"
""",
    )
    return source


def enable_upstream_fix(prepared: str) -> str:
    """Enable the PyTorch 2.13 opt-in behavior without changing the workload."""
    return replace_once(
        prepared,
        "        offload_policy=offload_policy\n    )\n\n\ndef apply_ddp(\n",
        "        offload_policy=offload_policy\n"
        "    )\n"
        "    # PyTorch 2.12 fix is opt-in, analogous to DDP "
        "find_unused_parameters.\n"
        "    model.set_reduce_scatter_unused_params(True)\n\n\n"
        "def apply_ddp(\n",
    )


def disable_random_output(prepared: str) -> str:
    """Create the same-version negative control by removing rank divergence."""
    return replace_once(
        prepared,
        "enable_random_output=(i < (num_stages - 1))",
        "enable_random_output=False",
    )


def read_source(source_path: Path | None) -> bytes:
    if source_path is not None:
        return source_path.read_bytes()
    with urlopen(SOURCE_URL, timeout=30) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--enable-upstream-fix", action="store_true")
    control.add_argument("--disable-random-output", action="store_true")
    args = parser.parse_args()

    raw = read_source(args.source)
    observed_hash = hashlib.sha256(raw).hexdigest()
    if observed_hash != SOURCE_SHA256:
        raise SystemExit(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, "
            f"observed {observed_hash}"
        )

    prepared = prepare_source(raw.decode("utf-8"))
    expected_prepared_hash = PREPARED_SHA256
    if args.enable_upstream_fix:
        prepared = enable_upstream_fix(prepared)
        expected_prepared_hash = FIXED_PREPARED_SHA256
    elif args.disable_random_output:
        prepared = disable_random_output(prepared)
        expected_prepared_hash = NO_DIVERGENCE_PREPARED_SHA256
    compile(prepared, str(args.output), "exec")
    prepared_bytes = prepared.encode("utf-8")
    prepared_hash = hashlib.sha256(prepared_bytes).hexdigest()
    if prepared_hash != expected_prepared_hash:
        raise SystemExit(
            f"prepared SHA-256 mismatch: expected {expected_prepared_hash}, "
            f"observed {prepared_hash}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(prepared_bytes)
    temporary.replace(args.output)
    print(
        f"prepared {args.output} from source {SOURCE_SHA256}; "
        f"prepared SHA-256 {expected_prepared_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
