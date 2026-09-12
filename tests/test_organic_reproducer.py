import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "organic-hang" / "fetch_and_prepare_reproducer.py"
ORACLE_SCRIPT = SCRIPT.with_name("parse_detail_oracle.py")
FLIGHT_SCRIPT = SCRIPT.with_name("normalize_flight_recorder.py")
STACK_SCRIPT = SCRIPT.with_name("stack_thread_join.py")
VERIFIER_SCRIPT = SCRIPT.with_name("verify_organic_results.py")
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "organic-hang"


def load_preparer():
    spec = importlib.util.spec_from_file_location("organic_preparer", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load organic reproducer preparer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oracle_parser():
    spec = importlib.util.spec_from_file_location("organic_oracle", ORACLE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load organic oracle parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OrganicReproducerPreparationTest(unittest.TestCase):
    def test_declared_source_identity_is_frozen(self) -> None:
        module = load_preparer()
        self.assertEqual(64, len(module.SOURCE_SHA256))
        self.assertEqual(64, len(module.PREPARED_SHA256))
        self.assertIn("fbd907f905a5e7ab", module.SOURCE_URL)

    def test_prepare_source_fails_closed_when_anchor_is_missing(self) -> None:
        module = load_preparer()
        with self.assertRaisesRegex(ValueError, "source anchor"):
            module.prepare_source("import os\n")

    def test_fixed_control_is_explicit_and_fail_closed(self) -> None:
        module = load_preparer()
        prepared = "        offload_policy=offload_policy\n    )\n\n\ndef apply_ddp(\n"
        fixed = module.enable_upstream_fix(prepared)
        self.assertIn("set_reduce_scatter_unused_params(True)", fixed)
        with self.assertRaisesRegex(ValueError, "source anchor"):
            module.enable_upstream_fix("def apply_ddp(\n")

    def test_same_version_negative_control_changes_only_random_output(self) -> None:
        module = load_preparer()
        prepared = "enable_random_output=(i < (num_stages - 1))"
        controlled = module.disable_random_output(prepared)
        self.assertEqual("enable_random_output=False", controlled)
        with self.assertRaisesRegex(ValueError, "source anchor"):
            module.disable_random_output("enable_random_output=True")

    def test_prepare_source_changes_only_declared_controls(self) -> None:
        module = load_preparer()
        source = (
            """import os
import abc
ModuleT = TypeVar('ModuleT', bound=nn.Module)
    global_rank = int(os.environ[GLOBAL_RANK_ENV_VAR])
"""
            + module.DEBUG_BLOCK
            + """
        timeout=datetime.timedelta(60)
    world_device_mesh = parallel_dims.build_mesh(device_type='cuda')
    os.environ[MASTER_PORT_ENV_VAR] = '8000'
        total_training_steps=200,
        debug=True
"""
        )
        prepared = module.prepare_source(source)
        self.assertIn("authorize_observer_from_env()", prepared)
        self.assertIn("DFX_ORGANIC_TIMEOUT_S", prepared)
        self.assertIn("DFX_ORGANIC_MASTER_PORT", prepared)
        self.assertIn("DFX_ORGANIC_TRAINING_STEPS", prepared)
        self.assertIn("DFX_ORGANIC_DEBUG_DETAIL", prepared)
        self.assertIn("DFX_STATE_DIR", prepared)
        self.assertIn("_set_pg_timeout(group_timeout, process_group)", prepared)
        self.assertIn("torch.autograd.set_detect_anomaly(False)", prepared)
        self.assertNotIn("torch.autograd.set_detect_anomaly(True)", prepared)
        self.assertNotIn("datetime.timedelta(60)", prepared)
        self.assertNotIn("MASTER_PORT_ENV_VAR] = '8000'", prepared)

    def test_detail_oracle_parser_extracts_structured_primary(self) -> None:
        module = load_oracle_parser()
        result = module.parse_oracle_text(
            "RuntimeError: Detected mismatch between collectives on ranks. "
            "Rank 2 is running collective: CollectiveFingerPrint("
            "SequenceNumber=19, OpType=_REDUCE_SCATTER_BASE, "
            "TensorShape=[1024], TensorDtypes=[Float]), but Rank 0 is "
            "running collective: CollectiveFingerPrint("
            "SequenceNumber=19, OpType=_REDUCE_SCATTER_BASE, "
            "TensorShape=[960], TensorDtypes=[Float])"
        )
        self.assertTrue(result["mismatch_detected"])
        self.assertEqual(1, len(result["records"]))
        self.assertEqual(19, result["primary_divergence"]["sequence_number"])
        self.assertEqual(
            "_REDUCE_SCATTER_BASE", result["primary_divergence"]["operation"]
        )
        self.assertEqual([0, 2], result["primary_divergence"]["group_members"])
        self.assertFalse(result["raw_persisted"])
        self.assertEqual(64, len(result["normalized_fingerprint"]))

    def test_detail_parser_recognizes_monitored_barrier(self) -> None:
        module = load_oracle_parser()
        result = module.parse_oracle_text(
            "ProcessGroupWrapper: Monitored Barrier encountered error "
            "running collective"
        )
        self.assertFalse(result["mismatch_detected"])
        self.assertEqual(1, result["monitored_barrier_error_count"])

    def test_detail_parser_recognizes_gloo_barrier_variant(self) -> None:
        module = load_oracle_parser()
        result = module.parse_oracle_text(
            "[Rank 0]: Ranks 1 failed to pass monitoredBarrier in 5000 ms"
        )
        self.assertFalse(result["mismatch_detected"])
        self.assertEqual(1, result["monitored_barrier_error_count"])

    def test_detail_parser_handles_nested_tensor_options(self) -> None:
        module = load_oracle_parser()
        result = module.parse_oracle_text(
            "Detected mismatch between collectives on ranks. Rank 1 is running "
            "collective: CollectiveFingerPrint(SequenceNumber=0, "
            "OpType=ALLREDUCE, TensorShape=[1024], TensorDtypes=Float, "
            "TensorDeviceTypes=TensorOptions(dtype=float, device=cpu)), but "
            "Rank 0 is running collective: CollectiveFingerPrint("
            "SequenceNumber=0, OpType=ALLREDUCE, TensorShape=[960], "
            "TensorDtypes=Float, TensorDeviceTypes=TensorOptions(dtype=float, "
            "device=cpu)).Collectives differ in the following aspects"
        )
        self.assertTrue(result["mismatch_detected"])
        self.assertEqual(0, result["primary_divergence"]["sequence_number"])

    def test_detail_parser_fails_closed_on_degraded_fingerprint(self) -> None:
        module = load_oracle_parser()
        result = module.parse_oracle_text(
            "Detected mismatch between collectives on ranks. Rank 1 is running "
            "collective: CollectiveFingerPrint(SequenceNumber=0, "
            "OpType=ALLREDUCE, TensorShape=[2, 2], TensorDtypes=Float), but "
            "Rank 0 is running collective: CollectiveFingerPrint(OpType=GATHER)"
        )
        self.assertTrue(result["mismatch_detected"])
        self.assertIsNone(result["primary_divergence"])

    def test_genuine_gloo_detail_fixtures(self) -> None:
        module = load_oracle_parser()
        valid = module.parse_oracle_text(
            (FIXTURE_DIR / "detail-long.txt").read_text(encoding="utf-8")
        )
        degraded = module.parse_oracle_text(
            (FIXTURE_DIR / "detail-degraded.txt").read_text(encoding="utf-8")
        )
        barrier = module.parse_oracle_text(
            (FIXTURE_DIR / "detail-barrier.txt").read_text(encoding="utf-8")
        )
        self.assertEqual("ALLREDUCE", valid["primary_divergence"]["operation"])
        self.assertTrue(degraded["mismatch_detected"])
        self.assertIsNone(degraded["primary_divergence"])
        self.assertEqual(1, barrier["monitored_barrier_error_count"])

    def test_detail_parser_handles_short_form_without_separator(self) -> None:
        module = load_oracle_parser()
        result = module.parse_oracle_text(
            "Detected mismatch between collectives on ranks. Rank 0 is running "
            "collective: CollectiveFingerPrint(SequenceNumber=7OpType=ALLREDUCE), "
            "but Rank 1 is running collective: CollectiveFingerPrint("
            "SequenceNumber=7OpType=ALLREDUCE)"
        )
        self.assertTrue(result["mismatch_detected"])
        self.assertEqual("ALLREDUCE", result["primary_divergence"]["operation"])

    def test_flight_normalizer_uses_group_and_sequence_key(self) -> None:
        module = load_module(FLIGHT_SCRIPT, "organic_flight")
        artifacts = []
        for rank, shape in ((0, 960), (1, 1024)):
            artifacts.append(
                {
                    "rank": rank,
                    "pg_config": {"3": {"ranks": [0, 1]}},
                    "entries": [
                        {
                            "pg_id": "3",
                            "is_p2p": False,
                            "collective_seq_id": 18,
                            "p2p_seq_id": 0,
                            "record_id": 1,
                            "profiling_name": "nccl:_reduce_scatter_base",
                            "input_sizes": [[7]],
                            "input_dtypes": ["BFloat16"],
                            "output_sizes": [[7]],
                            "output_dtypes": ["BFloat16"],
                            "state": "completed",
                            "thread_id": 20 + rank,
                            "thread_name": "pt_autograd_0",
                        },
                        {
                            "pg_id": "3",
                            "is_p2p": False,
                            "collective_seq_id": 19,
                            "p2p_seq_id": 0,
                            "record_id": 2,
                            "profiling_name": "nccl:_reduce_scatter_base",
                            "input_sizes": [[shape]],
                            "input_dtypes": ["BFloat16"],
                            "output_sizes": [[shape // 2]],
                            "output_dtypes": ["BFloat16"],
                            "state": "scheduled",
                            "thread_id": 20 + rank,
                            "thread_name": "pt_autograd_0",
                        },
                    ],
                }
            )
        result = module.normalize_rank_artifacts(artifacts)
        self.assertEqual(19, result["primary_divergence"]["sequence_number"])
        self.assertEqual([0, 1], result["primary_divergence"]["group_members"])
        self.assertEqual(2, result["completed_collective_entries"])

    def test_flight_normalizer_does_not_collide_process_groups(self) -> None:
        module = load_module(FLIGHT_SCRIPT, "organic_flight_groups")
        artifacts = []
        for rank in (0, 1, 2, 3):
            group = "a" if rank < 2 else "b"
            members = [0, 1] if rank < 2 else [2, 3]
            artifacts.append(
                {
                    "rank": rank,
                    "pg_config": {group: {"ranks": members}},
                    "entries": [
                        {
                            "pg_id": group,
                            "is_p2p": False,
                            "collective_seq_id": 1,
                            "p2p_seq_id": 0,
                            "record_id": 0,
                            "profiling_name": "nccl:all_reduce",
                            "input_sizes": [[8 if rank != 1 else 16]],
                            "input_dtypes": ["Float"],
                            "output_sizes": [[8]],
                            "output_dtypes": ["Float"],
                            "state": "scheduled",
                        }
                    ],
                }
            )
        result = module.normalize_rank_artifacts(artifacts)
        self.assertEqual([0, 1], result["primary_divergence"]["group_members"])

    def test_flight_normalizer_uses_global_group_id_from_real_pickle_shape(
        self,
    ) -> None:
        module = load_module(FLIGHT_SCRIPT, "organic_flight_real_pg")
        artifacts = []
        for rank, local_pg_id, global_pg_id, shape in (
            (0, 2, 3, 1024),
            (1, 1, 3, 960),
        ):
            artifacts.append(
                {
                    "rank": rank,
                    "pg_config": {
                        str(global_pg_id): {
                            "name": str(global_pg_id),
                            "desc": "mesh_pp",
                            "ranks": "[0, 1]",
                        }
                    },
                    "entries": [
                        {
                            "pg_id": local_pg_id,
                            "process_group": (str(global_pg_id), "mesh_pp"),
                            "is_p2p": False,
                            "collective_seq_id": 4,
                            "p2p_seq_id": 0,
                            "record_id": 5,
                            "profiling_name": "nccl:_reduce_scatter_base",
                            "input_sizes": [[shape]],
                            "input_dtypes": ["Float"],
                            "output_sizes": [[shape // 2]],
                            "output_dtypes": ["Float"],
                            "state": "scheduled",
                            "thread_id": str(100 + rank),
                            "thread_name": "pt_autograd_0",
                        }
                    ],
                }
            )
        result = module.normalize_rank_artifacts(artifacts)
        self.assertEqual(4, result["primary_divergence"]["sequence_number"])
        self.assertEqual([0, 1], result["primary_divergence"]["group_members"])

    def test_flight_normalizer_ignores_legal_p2p_send_recv_pair(self) -> None:
        module = load_module(FLIGHT_SCRIPT, "organic_flight_p2p")
        artifacts = []
        for rank, operation in ((0, "send"), (1, "recv")):
            artifacts.append(
                {
                    "rank": rank,
                    "pg_config": {"3": {"ranks": "[0, 1]"}},
                    "entries": [
                        {
                            "pg_id": rank + 1,
                            "process_group": ("3", "mesh_pp"),
                            "is_p2p": True,
                            "collective_seq_id": 2,
                            "p2p_seq_id": 7,
                            "record_id": 1,
                            "profiling_name": f"nccl:{operation}",
                            "input_sizes": [[8]],
                            "input_dtypes": ["Float"],
                            "output_sizes": [],
                            "output_dtypes": [],
                            "state": "scheduled",
                        }
                    ],
                }
            )
        result = module.normalize_rank_artifacts(artifacts)
        self.assertIsNone(result["primary_divergence"])
        self.assertEqual([], result["secondary_divergences"])

    def test_flight_normalizer_ignores_coalesced_wrappers(self) -> None:
        module = load_module(FLIGHT_SCRIPT, "organic_flight_coalesced")
        artifacts = []
        for rank in (0, 1):
            artifacts.append(
                {
                    "rank": rank,
                    "pg_config": {"3": {"ranks": [0, 1]}},
                    "entries": [
                        {
                            "pg_id": "3",
                            "is_p2p": False,
                            "collective_seq_id": 2,
                            "p2p_seq_id": 0,
                            "record_id": 1,
                            "profiling_name": "nccl:coalesced",
                            "input_sizes": [[8 if rank == 0 else 16]],
                            "input_dtypes": ["Float"],
                            "output_sizes": [],
                            "output_dtypes": [],
                            "state": "scheduled",
                        }
                    ],
                }
            )
        result = module.normalize_rank_artifacts(artifacts)
        self.assertIsNone(result["primary_divergence"])

    def test_verifier_matches_detail_local_ranks_to_fr_global_group(self) -> None:
        module = load_module(VERIFIER_SCRIPT, "organic_verifier_semantics")
        oracle = {
            "sequence_number": 4,
            "operation": "_REDUCE_SCATTER_BASE",
            "group_members": [0, 1],
            "is_p2p": False,
            "input_shapes_by_rank": [
                {"rank": 0, "shapes": [[1024]]},
                {"rank": 1, "shapes": [[960]]},
            ],
            "input_dtypes_by_rank": [
                {"rank": 0, "dtypes": ["Float"]},
                {"rank": 1, "dtypes": ["Float"]},
            ],
        }
        recorder = {
            **oracle,
            "sequence_number": 5,
            "group_members": [0, 2],
            "input_shapes_by_rank": [
                {"rank": 0, "shapes": [[1024]]},
                {"rank": 2, "shapes": [[960]]},
            ],
            "input_dtypes_by_rank": [
                {"rank": 0, "dtypes": ["Float"]},
                {"rank": 2, "dtypes": ["Float"]},
            ],
        }
        module._assert_primary_matches(oracle, recorder)

    def test_verifier_rejects_same_size_wrong_global_group(self) -> None:
        module = load_module(VERIFIER_SCRIPT, "organic_verifier_group")
        oracle = {
            "sequence_number": 4,
            "operation": "_REDUCE_SCATTER_BASE",
            "group_members": [0, 1],
            "is_p2p": False,
            "input_shapes_by_rank": [
                {"rank": 0, "shapes": [[1024]]},
                {"rank": 1, "shapes": [[960]]},
            ],
            "input_dtypes_by_rank": [
                {"rank": 0, "dtypes": ["Float Float"]},
                {"rank": 1, "dtypes": ["Float Float"]},
            ],
        }
        wrong_group = {
            **oracle,
            "group_members": [2, 3],
            "input_shapes_by_rank": [
                {"rank": 2, "shapes": [[1024]]},
                {"rank": 3, "shapes": [[960]]},
            ],
            "input_dtypes_by_rank": [
                {"rank": 2, "dtypes": ["Float"]},
                {"rank": 3, "dtypes": ["Float"]},
            ],
        }
        with self.assertRaises(AssertionError):
            module._assert_primary_matches(oracle, wrong_group)

    def test_verifier_rejects_stable_but_wrong_primary(self) -> None:
        module = load_module(VERIFIER_SCRIPT, "organic_verifier")
        oracle = {
            "sequence_number": 19,
            "operation": "_REDUCE_SCATTER_BASE",
            "group_members": [0, 1],
            "is_p2p": False,
            "input_shapes_by_rank": [
                {"rank": 0, "shapes": [[960]]},
                {"rank": 1, "shapes": [[1024]]},
            ],
        }
        wrong = {
            **oracle,
            "operation": "SEND",
        }
        with self.assertRaises(AssertionError):
            module._assert_primary_matches(oracle, wrong)

    def test_stack_join_uses_os_tid_and_exact_path(self) -> None:
        module = load_module(STACK_SCRIPT, "organic_stack")
        joined = module.match_issuing_thread(
            flight_entry={"thread_id": 44},
            py_spy_threads=[{"os_thread_id": 44, "frames": []}],
            task_names={44: "pt_autograd_0"},
        )
        self.assertEqual("matched", joined["outcome"])
        frame = module.first_project_frame(
            [
                {"filename": str(SCRIPT), "name": "run_training", "line": 10},
                {"filename": "site-packages/x.py", "name": "x", "line": 1},
            ],
            prepared_target=SCRIPT,
        )
        self.assertEqual(SCRIPT.name, frame["file"])


if __name__ == "__main__":
    unittest.main()
