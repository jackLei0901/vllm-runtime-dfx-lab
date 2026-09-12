"""Verify the published, derived-only organic-hang evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_OPERATION = "_REDUCE_SCATTER_BASE"
EXPECTED_GLOBAL_DP_GROUP = [0, 2]
EXPECTED_SHAPES = [[[1024]], [[960]]]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def values(divergence: dict, field: str, value_field: str) -> list[object]:
    return sorted(
        (item[value_field] for item in divergence[field]),
        key=lambda value: json.dumps(value, sort_keys=True),
    )


def dtype_families(divergence: dict) -> list[str]:
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


def main() -> int:
    oracles = [
        load(ROOT / f"detail-oracle-{trial}" / "oracle.json") for trial in range(1, 4)
    ]
    recorders = [
        load(ROOT / f"automatic-hang-{trial}" / "flight-recorder.json")
        for trial in range(1, 4)
    ]

    assert len({item["normalized_fingerprint"] for item in oracles}) == 1
    assert len({item["normalized_fingerprint"] for item in recorders}) == 1

    oracle = oracles[0]["primary_divergence"]
    recorder = recorders[0]["primary_divergence"]
    assert oracle is not None and recorder is not None
    assert oracle["operation"] == recorder["operation"] == EXPECTED_OPERATION
    assert recorder["group_members"] == EXPECTED_GLOBAL_DP_GROUP
    assert len(oracle["group_members"]) == len(recorder["group_members"]) == 2
    assert values(oracle, "input_shapes_by_rank", "shapes") == values(
        recorder, "input_shapes_by_rank", "shapes"
    )
    assert values(oracle, "input_shapes_by_rank", "shapes") == EXPECTED_SHAPES
    assert dtype_families(oracle) == dtype_families(recorder) == ["Float", "Float"]
    assert all(item["completed_collective_entries"] > 0 for item in recorders)

    retained_lifecycle = [
        ROOT / "detail-oracle-2" / "lifecycle.json",
        ROOT / "detail-oracle-3" / "lifecycle.json",
        *(ROOT / f"automatic-hang-{trial}" / "lifecycle.json" for trial in range(1, 4)),
    ]
    for path in retained_lifecycle:
        lifecycle = load(path)
        assert lifecycle["cleanup"]["pid_start_times_verified"] is True
        assert lifecycle["cleanup"]["no_orphans"] is True

    control = load(ROOT / "cross-version-opt-in-control-summary.json")
    assert control["status"] == "fixed_control_blocked"
    assert control["fixed_control"]["trial_count"] == 3
    assert control["fixed_control"]["exit_codes"] == [1, 1, 1]
    assert control["fixed_control"]["orphan_counts"] == [0, 0, 0]

    print(
        "PASS (3 stable DETAIL records match 3 stable Flight Recorder records; "
        "expected DP group and dtype family confirmed; retained cleanup records pass)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
