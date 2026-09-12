"""Verify a completed organic FSDP2 hang campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCE_SHA256 = "f23a41bebca5bcac51c6433ecc4a837fa3bbd1b5fd552c6701373619fafe0654"
PREPARED_SHA256 = "47fce815563b5ce8e94f93e18f1184b73fa0382ccd3934b7df02a90590c0a5a2"
FIXED_PREPARED_SHA256 = (
    "430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1"
)
EXPECTED_RECORDER_PRIMARY_GROUP = [0, 2]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _shape_map(divergence: dict) -> dict[int, list[list[int]]]:
    return {
        int(item["rank"]): item["shapes"] for item in divergence["input_shapes_by_rank"]
    }


def _value_multiset(divergence: dict, field: str, value_field: str) -> list[str]:
    return sorted(_canonical(item[value_field]) for item in divergence[field])


def _dtype_family_multiset(divergence: dict) -> list[str]:
    families = []
    for item in divergence["input_dtypes_by_rank"]:
        tokens = {
            token
            for value in item["dtypes"]
            for token in str(value).replace(",", " ").split()
        }
        assert len(tokens) == 1, f"non-uniform dtype representation: {tokens}"
        families.append(tokens.pop())
    return sorted(families)


def _assert_primary_matches(oracle: dict, recorder: dict) -> None:
    assert recorder["operation"] == oracle["operation"]
    assert recorder["is_p2p"] is False
    assert oracle["is_p2p"] is False
    # This case diverges in the FSDP group for pipeline stage 0. Checking the
    # concrete global group prevents an unrelated two-rank PP/DP group with a
    # coincidentally equal payload from satisfying the case-specific oracle.
    assert recorder["group_members"] == EXPECTED_RECORDER_PRIMARY_GROUP
    assert len(recorder["group_members"]) == len(oracle["group_members"])
    # DETAIL identifies ranks within its wrapped process group, whereas Flight
    # Recorder retains global group membership. Compare the semantic payload,
    # not rank labels or sequence counters from different identity spaces.
    assert _value_multiset(
        recorder, "input_shapes_by_rank", "shapes"
    ) == _value_multiset(oracle, "input_shapes_by_rank", "shapes")
    # DETAIL combines input/output dtype labels ("Float Float") while the
    # normalized Flight Recorder primary stores input dtype labels ("Float").
    # Require a uniform, equal dtype family without pretending these arrays
    # have the same source shape.
    assert _dtype_family_multiset(recorder) == _dtype_family_multiset(oracle)


def _assert_arm_contract(summary: dict) -> None:
    arm = summary["arm"]
    expected_base = "2.13.0" if arm == "fixed_control" else "2.11.0"
    assert summary["runtime"]["torch_base_version"] == expected_base
    assert summary["runtime"]["torch"].startswith(expected_base)
    assert summary["preflight"]["torch"] == summary["runtime"]["torch"]
    assert summary["preflight"]["driver"] == summary["runtime"]["driver"]
    expected_prepared = (
        FIXED_PREPARED_SHA256 if arm == "fixed_control" else PREPARED_SHA256
    )
    assert summary["source"] == {
        "upstream_sha256": SOURCE_SHA256,
        "prepared_sha256": expected_prepared,
    }
    assert summary["preflight"]["prepared_sha256"] == expected_prepared
    assert summary["configuration"]["training_steps"] == 200
    assert summary["configuration"]["anomaly_detection"] is False
    assert summary["cleanup"]["pid_start_times_verified"] is True
    assert summary["cleanup"]["no_orphans"] is True
    assert summary["raw_persisted"] is False

    if arm == "detail_oracle":
        assert summary["configuration"]["detail_enabled"] is True
        assert summary["configuration"]["flight_recorder_trigger"] == "none"
        assert summary["trigger"]["kind"] == "detail_wrapper"
    elif arm == "automatic_hang":
        assert summary["configuration"]["detail_enabled"] is False
        assert (
            summary["configuration"]["flight_recorder_trigger"] == "automatic_timeout"
        )
        assert summary["trigger"] == {
            "kind": "automatic_timeout",
            "timeout_seconds": 60,
            "debug_pipe_used": False,
            "py_spy_started_before_artifact": False,
        }
    elif arm == "post_gate_stack":
        assert summary["configuration"]["detail_enabled"] is False
        assert summary["configuration"]["flight_recorder_trigger"] == "manual_dump"
        assert summary["trigger"]["kind"] == "manual_dump"
        assert summary["trigger"]["debug_pipe_used"] is True
    else:
        assert summary["configuration"]["detail_enabled"] is False
        assert summary["configuration"]["flight_recorder_trigger"] == "none"
        assert summary["trigger"]["kind"] == "normal_completion"


def verify(root: Path, schema_path: Path) -> int:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load(schema_path))
    paths_by_arm = {
        "detail_oracle": sorted(root.glob("detail-oracle-*/summary.json")),
        "automatic_hang": sorted(root.glob("automatic-hang-*/summary.json")),
        "post_gate_stack": sorted(root.glob("post-gate-stack-*/summary.json")),
        "fixed_control": sorted(root.glob("fixed-control-*/summary.json")),
    }
    assert len(paths_by_arm["detail_oracle"]) == 3, "expected three DETAIL trials"
    assert len(paths_by_arm["automatic_hang"]) == 3, "expected three hang trials"
    assert len(paths_by_arm["fixed_control"]) == 3, "expected three fixed trials"
    assert len(paths_by_arm["post_gate_stack"]) in (0, 3), (
        "Arm C is absent or three trials"
    )

    summaries_by_arm: dict[str, list[dict]] = {}
    for arm, paths in paths_by_arm.items():
        summaries_by_arm[arm] = []
        for path in paths:
            summary = load(path)
            errors = sorted(
                validator.iter_errors(summary), key=lambda error: list(error.path)
            )
            assert not errors, f"{path}: {errors[0].message if errors else ''}"
            assert summary["arm"] == arm, path
            _assert_arm_contract(summary)
            summaries_by_arm[arm].append(summary)

    oracle_divergences = []
    for summary in summaries_by_arm["detail_oracle"]:
        oracle = summary["outcome"]["oracle"]
        assert summary["outcome"]["classification"] == "collective_mismatch"
        assert oracle["mismatch_detected"] is True
        assert oracle["record_count"] >= 1
        divergence = oracle["primary_divergence"]
        assert divergence is not None
        assert divergence["operation"] == "_REDUCE_SCATTER_BASE"
        assert divergence["is_p2p"] is False
        assert len(set(map(_canonical, _shape_map(divergence).values()))) > 1
        oracle_divergences.append(divergence)
    assert len({_canonical(value) for value in oracle_divergences}) == 1, (
        "DETAIL primary divergence is not stable"
    )
    oracle_divergence = oracle_divergences[0]

    hang_divergences = []
    for summary in summaries_by_arm["automatic_hang"]:
        recorder = summary["outcome"]["flight_recorder"]
        assert summary["outcome"]["classification"] == "hang"
        assert recorder["artifact_count"] == 4
        assert recorder["completed_collective_entries"] > 0, (
            "hang is not distinguished from setup failure"
        )
        assert summary["cleanup"]["rank_pids_recorded"] == 4
        divergence = recorder["primary_divergence"]
        assert divergence is not None
        _assert_primary_matches(oracle_divergence, divergence)
        hang_divergences.append(divergence)
    assert len({_canonical(value) for value in hang_divergences}) == 1, (
        "hang primary divergence is unstable"
    )

    for summary in summaries_by_arm["fixed_control"]:
        assert summary["outcome"]["classification"] == "completed"
        assert summary["outcome"]["process_exit_code"] == 0
        assert summary["outcome"]["completed_training_steps"] == 200

    for summary in summaries_by_arm["post_gate_stack"]:
        stack = summary["outcome"]["external_stack"]
        assert stack["expected_incremental_value"] is False

    print(
        "PASS (3 DETAIL oracle trials match 3 automatic Flight Recorder trials; "
        "3 cross-version opt-in controls complete)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("organic-hang-result-v1.schema.json"),
    )
    args = parser.parse_args()
    return verify(args.result_root, args.schema)


if __name__ == "__main__":
    raise SystemExit(main())
