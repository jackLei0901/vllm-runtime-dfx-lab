import json
import unittest
from copy import deepcopy
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:  # The project declares jsonschema in its dev extra.
    Draft202012Validator = None


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    ROOT / "experiments" / "organic-hang" / "organic-hang-result-v1.schema.json"
)
SOURCE_SHA256 = "f23a41bebca5bcac51c6433ecc4a837fa3bbd1b5fd552c6701373619fafe0654"
PREPARED_SHA256 = "47fce815563b5ce8e94f93e18f1184b73fa0382ccd3934b7df02a90590c0a5a2"
FIXED_PREPARED_SHA256 = (
    "430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1"
)


def valid_summary() -> dict:
    return {
        "schema_version": 1,
        "case_id": "PYTORCH-FSDP-158719",
        "protocol_revision": "2026-09-12.6",
        "arm": "automatic_hang",
        "trial": 1,
        "source": {
            "upstream_sha256": SOURCE_SHA256,
            "prepared_sha256": PREPARED_SHA256,
        },
        "runtime": {
            "python": "3.12.3",
            "torch": "2.11.0+cu130",
            "torch_base_version": "2.11.0",
            "cuda": "13.0",
            "nccl": [2, 28, 9],
            "driver": "580.82.07",
            "gpu_model": "test-gpu",
        },
        "preflight": {
            "status": "PASS",
            "torch": "2.11.0+cu130",
            "driver": "580.82.07",
            "prepared_sha256": PREPARED_SHA256,
            "visible_gpu_count": 4,
            "py_spy_available": True,
        },
        "configuration": {
            "training_steps": 200,
            "anomaly_detection": False,
            "detail_enabled": False,
            "flight_recorder_trigger": "automatic_timeout",
        },
        "topology": {
            "backend": "nccl",
            "world_size": 4,
            "pp_size": 2,
            "dp_shard_size": 2,
        },
        "trigger": {
            "kind": "automatic_timeout",
            "timeout_seconds": 60,
            "debug_pipe_used": False,
            "py_spy_started_before_artifact": False,
        },
        "outcome": {
            "classification": "hang",
            "process_exit_code": None,
            "completed_training_steps": 1,
            "oracle": None,
            "flight_recorder": {
                "artifact_count": 4,
                "primary_divergence": {
                    "sequence_number": 19,
                    "operation": "_REDUCE_SCATTER_BASE",
                    "group_members": [0, 1],
                    "is_p2p": False,
                    "input_shapes_by_rank": [
                        {"rank": 0, "shapes": [[960]]},
                        {"rank": 1, "shapes": [[1024]]},
                    ],
                    "input_dtypes_by_rank": [
                        {"rank": 0, "dtypes": ["BFloat16"]},
                        {"rank": 1, "dtypes": ["BFloat16"]},
                    ],
                },
                "secondary_divergence_count": 2,
                "completed_collective_entries": 6,
                "normalized_fingerprint": "c" * 64,
                "raw_persisted": False,
            },
            "external_stack": None,
        },
        "cleanup": {
            "rank_pids_recorded": 4,
            "pid_start_times_verified": True,
            "no_orphans": True,
        },
        "raw_persisted": False,
    }


@unittest.skipIf(Draft202012Validator is None, "jsonschema dev extra is unavailable")
class OrganicHangSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert Draft202012Validator is not None
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_minimal_automatic_hang_summary_is_valid(self) -> None:
        self.assertEqual([], list(self.validator.iter_errors(valid_summary())))

    def test_unknown_top_level_field_is_rejected(self) -> None:
        summary = deepcopy(valid_summary())
        summary["raw_log"] = "must not be accepted"
        self.assertTrue(list(self.validator.iter_errors(summary)))

    def test_fixed_control_requires_fixed_prepared_source(self) -> None:
        summary = valid_summary()
        summary["arm"] = "fixed_control"
        summary["runtime"]["torch"] = "2.13.0+cu130"
        summary["runtime"]["torch_base_version"] = "2.13.0"
        summary["preflight"]["torch"] = "2.13.0+cu130"
        summary["source"]["prepared_sha256"] = FIXED_PREPARED_SHA256
        summary["preflight"]["prepared_sha256"] = FIXED_PREPARED_SHA256
        self.assertEqual([], list(self.validator.iter_errors(summary)))

        summary["source"]["prepared_sha256"] = PREPARED_SHA256
        self.assertTrue(list(self.validator.iter_errors(summary)))

    def test_affected_arm_rejects_fixed_prepared_source(self) -> None:
        summary = valid_summary()
        summary["source"]["prepared_sha256"] = FIXED_PREPARED_SHA256
        summary["preflight"]["prepared_sha256"] = FIXED_PREPARED_SHA256
        self.assertTrue(list(self.validator.iter_errors(summary)))


if __name__ == "__main__":
    unittest.main()
