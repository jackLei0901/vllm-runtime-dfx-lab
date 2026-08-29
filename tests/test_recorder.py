import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dfxlab.recorder import IncidentRecorder
from dfxlab.schema import Sample


def sample(
    *, healthy: bool = True, alive: bool = True, metrics: dict[str, float] | None = None
) -> Sample:
    return Sample(
        timestamp="2026-01-01T00:00:00+00:00",
        monotonic_seconds=1.0,
        process={"pid": 1, "alive": alive},
        health={"ok": healthy, "status": 200 if healthy else None},
        metrics=metrics or {},
    )


class RecorderTest(unittest.TestCase):
    def make_recorder(self, directory: Path) -> IncidentRecorder:
        with patch("dfxlab.recorder.environment_snapshot", return_value={"test": True}):
            return IncidentRecorder(
                "http://127.0.0.1:8000",
                directory,
                pid=1,
                history_size=2,
                unhealthy_samples=2,
                preemption_delta=5,
            )

    def test_bounded_history_and_atomic_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            for index in range(3):
                payload = sample(metrics={"counter": float(index)}).to_dict()
                recorder.history.append(payload)
            path = recorder.capture("test", "warning")
            self.assertTrue(path.exists())
            self.assertEqual(len(recorder.history), 2)

    def test_process_exit_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            self.assertEqual(recorder.classify(sample(alive=False)), ("process_exit", "fatal"))

    def test_health_must_have_been_healthy_before_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            self.assertIsNone(recorder.classify(sample(healthy=False)))
            self.assertIsNone(recorder.classify(sample(healthy=True)))
            self.assertIsNone(recorder.classify(sample(healthy=False)))
            self.assertEqual(
                recorder.classify(sample(healthy=False)), ("health_lost", "fatal")
            )

    def test_pressure_and_preemption_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            self.assertEqual(
                recorder.classify(
                    sample(metrics={"vllm:kv_cache_usage_perc": 0.98})
                ),
                ("kv_pressure", "warning"),
            )
            self.assertIsNone(
                recorder.classify(
                    sample(metrics={"vllm:num_preemptions_total": 1.0})
                )
            )
            self.assertEqual(
                recorder.classify(
                    sample(metrics={"vllm:num_preemptions_total": 7.0})
                ),
                ("preemption_storm", "warning"),
            )


if __name__ == "__main__":
    unittest.main()

