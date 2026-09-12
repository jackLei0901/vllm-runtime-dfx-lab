import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "organic-hang" / "process_lifecycle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("organic_lifecycle", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load organic lifecycle module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_stat(root: Path, pid: int, start_time: int) -> None:
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    # Fields after comm begin at field 3; start time is field 22.
    suffix = ["S", *["0"] * 18, str(start_time), *["0"] * 4]
    (directory / "stat").write_text(
        f"{pid} (rank worker) " + " ".join(suffix), encoding="utf-8"
    )


class OrganicLifecycleTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "requires Linux /proc and process groups")
    def test_real_posix_process_group_cleanup(self) -> None:
        module = load_module()
        child_program = "import time; time.sleep(60)"
        parent_program = """
import os
import pathlib
import subprocess
import sys
import time

state = pathlib.Path(os.environ["DFX_TEST_STATE_DIR"])
children = [
    subprocess.Popen([sys.executable, "-c", sys.argv[1]]) for _ in range(4)
]
for rank, child in enumerate(children):
    target = state / f"rank-{rank}.pid"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(str(child.pid), encoding="utf-8")
    temporary.replace(target)
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir()
            environment = os.environ.copy()
            environment["DFX_TEST_STATE_DIR"] = str(state)
            parent, parent_identity = module.start_campaign_process(
                [sys.executable, "-c", parent_program, child_program], environment
            )
            ranks = module.wait_for_rank_identities(
                state,
                4,
                module.time.monotonic() + 5,
                parent.poll,
            )
            outcome = module.cleanup_process_group(
                parent, parent_identity, ranks, grace_seconds=2
            )
            self.assertTrue(outcome["no_orphans"])

    def test_stat_parser_handles_spaces_in_comm(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stat(root, 42, 123456)
            self.assertEqual(123456, module.read_start_time(42, root))

    def test_wait_binds_four_unique_rank_identities(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            state.mkdir()
            for rank, pid in enumerate((10, 11, 12, 13)):
                write_stat(root, pid, 100 + rank)
                (state / f"rank-{rank}.pid").write_text(str(pid), encoding="utf-8")
            identities = module.wait_for_rank_identities(
                state, 4, module.time.monotonic() + 1, lambda: None, root
            )
            self.assertEqual([10, 11, 12, 13], [item.pid for item in identities])

    def test_wall_clock_deadline_is_bounded(self) -> None:
        module = load_module()
        started = module.time.monotonic()
        result = module.wait_until_exit_or_deadline(
            lambda: None, started + 0.02, poll_interval_seconds=0.001
        )
        self.assertIsNone(result)
        self.assertLess(module.time.monotonic() - started, 0.2)

    def test_pid_reuse_is_never_signalled(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stat(root, 50, 100)
            identity = module.identify(50, root)
            write_stat(root, 50, 200)
            with patch.object(module.os, "kill") as kill:
                self.assertFalse(
                    module._signal_if_same(identity, module.TERM_SIGNAL, root)
                )
            kill.assert_not_called()

    def test_cleanup_skips_reused_parent_and_rank(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stat(root, 60, 100)
            write_stat(root, 61, 101)
            parent_identity = module.identify(60, root)
            rank_identity = module.identify(61, root)
            write_stat(root, 60, 200)
            write_stat(root, 61, 201)
            parent = Mock()
            parent.wait.return_value = 0
            with (
                patch.object(module.os, "kill") as kill,
                patch.object(module.os, "killpg", create=True) as killpg,
            ):
                outcome = module.cleanup_process_group(
                    parent,
                    parent_identity,
                    [rank_identity],
                    grace_seconds=0,
                    proc_root=root,
                )
            kill.assert_not_called()
            killpg.assert_not_called()
            self.assertTrue(outcome["no_orphans"])
            self.assertEqual([], outcome["remaining_tracked_pids"])

    def test_cleanup_escalates_matching_processes(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stat(root, 70, 100)
            write_stat(root, 71, 101)
            parent_identity = module.identify(70, root)
            rank_identity = module.identify(71, root)
            parent = Mock()
            parent.wait.side_effect = subprocess.TimeoutExpired("test", 0)
            with (
                patch.object(module.os, "kill") as kill,
                patch.object(module.os, "killpg", create=True) as killpg,
                patch.object(
                    module,
                    "identity_is_live",
                    side_effect=[
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                    ],
                ),
            ):
                outcome = module.cleanup_process_group(
                    parent,
                    parent_identity,
                    [rank_identity],
                    grace_seconds=0,
                    proc_root=root,
                )
            self.assertEqual(
                [
                    unittest.mock.call(71, module.TERM_SIGNAL),
                    unittest.mock.call(71, module.KILL_SIGNAL),
                ],
                kill.call_args_list,
            )
            self.assertEqual(
                [
                    unittest.mock.call(70, module.TERM_SIGNAL),
                    unittest.mock.call(70, module.KILL_SIGNAL),
                ],
                killpg.call_args_list,
            )
            self.assertTrue(outcome["no_orphans"])
