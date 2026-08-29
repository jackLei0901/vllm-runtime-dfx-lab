import tempfile
import unittest
from pathlib import Path

from dfxlab.faults import inject_signal


class FaultsTest(unittest.TestCase):
    def test_dry_run_records_without_signalling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            event = inject_signal(999999, "TERM", path, dry_run=True)
            self.assertEqual(event["signal"], "SIGTERM")
            self.assertTrue(path.exists())

    def test_unknown_signal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inject_signal(1, "NOT_A_SIGNAL", dry_run=True)


if __name__ == "__main__":
    unittest.main()

