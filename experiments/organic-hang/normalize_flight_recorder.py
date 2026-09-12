"""Normalize multi-process-group Flight Recorder records without clock joins."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any


def _members(pg_config: dict[str, Any], pg_id: str) -> list[int]:
    config = pg_config.get(str(pg_id), {})
    value = config.get("ranks", config.get("global_ranks", config.get("members")))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError(f"process group {pg_id} has no global-rank membership")
    return sorted(int(rank) for rank in value)


def _operation(entry: dict[str, Any]) -> str:
    name = str(entry.get("profiling_name", "unknown"))
    return name.rsplit(":", 1)[-1].upper()


def _process_group_id(entry: dict[str, Any]) -> str:
    """Return the globally meaningful process-group identifier.

    PyTorch 2.11 records ``pg_id`` as a rank-local index. The first element of
    ``process_group`` is the identifier used by ``pg_config`` and is stable
    across the ranks that belong to that group. JSON decoding changes the
    observed pickle tuple into a list, so accept both representations.
    """
    process_group = entry.get("process_group")
    if isinstance(process_group, (list, tuple)) and process_group:
        return str(process_group[0])
    return str(entry["pg_id"])


def normalize_rank_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a bounded divergence summary from decoded per-rank artifacts."""
    grouped: dict[tuple[tuple[int, ...], bool, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    completed_before = 0
    for artifact in artifacts:
        rank = int(artifact["rank"])
        pg_config = artifact["pg_config"]
        for entry in sorted(artifact["entries"], key=lambda value: value["record_id"]):
            operation = _operation(entry)
            # P2P peers legitimately record different operations (SEND versus
            # RECV), and COALESCED is a wrapper rather than one comparable
            # collective. Neither can define a cross-rank collective mismatch.
            if bool(entry.get("is_p2p", False)) or operation == "COALESCED":
                continue
            pg_id = _process_group_id(entry)
            members = _members(pg_config, pg_id)
            sequence = int(entry["collective_seq_id"])
            grouped[(tuple(members), False, sequence)].append(
                {
                    "rank": rank,
                    "operation": operation,
                    "input_sizes": entry.get("input_sizes", []),
                    "input_dtypes": entry.get("input_dtypes", []),
                    "output_sizes": entry.get("output_sizes", []),
                    "output_dtypes": entry.get("output_dtypes", []),
                    "state": entry.get("state", "unknown"),
                    "record_id": int(entry["record_id"]),
                    "thread_id": entry.get("thread_id"),
                    "thread_name": entry.get("thread_name"),
                }
            )
            if entry.get("state") == "completed":
                completed_before += 1

    primary = None
    secondary = []
    # This lexicographic traversal makes `primary` reproducible. It is not a
    # temporal ordering across process groups: record_id is rank-local and wall
    # clocks are intentionally excluded from the join contract.
    for (members, is_p2p, sequence), participants in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for participant in participants:
            by_rank[participant["rank"]].append(participant)
        present = sorted(by_rank)
        ambiguous_ranks = [rank for rank, values in by_rank.items() if len(values) != 1]
        if ambiguous_ranks:
            secondary.append(
                {
                    "sequence_number": sequence,
                    "group_members": list(members),
                    "present_ranks": present,
                    "is_p2p": is_p2p,
                    "reason": "ambiguous_rank_entries",
                }
            )
            continue
        participants = [by_rank[rank][0] for rank in present]
        signature_values = {
            (
                item["operation"],
                json.dumps(item["input_sizes"], sort_keys=True),
                json.dumps(item["input_dtypes"], sort_keys=True),
                json.dumps(item["output_sizes"], sort_keys=True),
                json.dumps(item["output_dtypes"], sort_keys=True),
            )
            for item in participants
        }
        all_members_present = present == list(members)
        if all_members_present and len(signature_values) > 1 and primary is None:
            operations = {item["operation"] for item in participants}
            primary = {
                "sequence_number": sequence,
                "operation": operations.pop() if len(operations) == 1 else "MIXED",
                "group_members": list(members),
                "is_p2p": is_p2p,
                "input_shapes_by_rank": [
                    {"rank": item["rank"], "shapes": item["input_sizes"]}
                    for item in sorted(participants, key=lambda value: value["rank"])
                ],
                "input_dtypes_by_rank": [
                    {"rank": item["rank"], "dtypes": item["input_dtypes"]}
                    for item in sorted(participants, key=lambda value: value["rank"])
                ],
            }
            continue
        all_pending = all(item["state"] != "completed" for item in participants)
        if not all_members_present or all_pending:
            secondary.append(
                {
                    "sequence_number": sequence,
                    "group_members": list(members),
                    "present_ranks": present,
                    "is_p2p": is_p2p,
                    "reason": (
                        "missing_member" if not all_members_present else "all_pending"
                    ),
                }
            )

    bounded = {
        "primary_divergence": primary,
        "secondary_divergences": secondary,
        "completed_collective_entries": completed_before,
    }
    canonical = json.dumps(bounded, sort_keys=True, separators=(",", ":"))
    return {
        **bounded,
        "normalized_fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
    }
