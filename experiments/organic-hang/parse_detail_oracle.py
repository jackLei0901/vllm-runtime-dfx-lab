"""Extract bounded, structured records from PyTorch DETAIL-mode stderr."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

MISMATCH_TEXT = "Detected mismatch between collectives on ranks"
BARRIER_TEXTS = (
    "ProcessGroupWrapper: Monitored Barrier encountered error",
    "failed to pass monitoredBarrier",
)
MESSAGE_START = re.compile(
    rf"(?={re.escape(MISMATCH_TEXT)}|"
    + "|".join(re.escape(value) for value in BARRIER_TEXTS)
    + r")"
)
FINGERPRINT = re.compile(
    r"Rank\s+(?P<rank>\d+).*?CollectiveFingerPrint\((?P<body>.*?)\)+"
    r"(?=\s*(?:,?\s*but\s+Rank|\.?\s*Collectives\s+differ|$))",
    re.DOTALL,
)
SEQUENCE = re.compile(r"SequenceNumber=(\d+)")
OPERATION = re.compile(r"OpType=([A-Z0-9_]+)")
SHAPE = re.compile(r"TensorShape=\[([^\]]*)\]")
DTYPE = re.compile(r"TensorDtypes?=(?:\[([^\]]*)\]|([^,\)]+))")


def _normalize_shape(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _messages(text: str) -> list[str]:
    starts = [match.start() for match in MESSAGE_START.finditer(text)]
    return [
        text[start : starts[index + 1] if index + 1 < len(starts) else len(text)]
        for index, start in enumerate(starts)
    ]


def _parse_collective_message(message: str) -> dict[str, Any] | None:
    participants = []
    for match in FINGERPRINT.finditer(message):
        body = match.group("body")
        sequence = SEQUENCE.search(body)
        operation = OPERATION.search(body)
        shape = SHAPE.search(body)
        dtype = DTYPE.search(body)
        participants.append(
            {
                "rank": int(match.group("rank")),
                "sequence_number": int(sequence.group(1)) if sequence else None,
                "operation": operation.group(1) if operation else "unknown",
                "input_shape": _normalize_shape(shape.group(1)) if shape else [],
                "input_dtype": (
                    next(value for value in dtype.groups() if value is not None).strip()
                    if dtype
                    else "unknown"
                ),
            }
        )
    if len(participants) < 2:
        return None
    return {"kind": "collective_mismatch", "participants": participants}


def _primary_divergence(record: dict[str, Any]) -> dict[str, Any] | None:
    participants = record["participants"]
    sequences = {item["sequence_number"] for item in participants}
    operations = {item["operation"] for item in participants}
    if None in sequences or len(sequences) != 1 or len(operations) != 1:
        return None
    return {
        "sequence_number": sequences.pop(),
        "operation": operations.pop(),
        "group_members": sorted(item["rank"] for item in participants),
        "is_p2p": False,
        "input_shapes_by_rank": [
            {"rank": item["rank"], "shapes": [item["input_shape"]]}
            for item in sorted(participants, key=lambda value: value["rank"])
        ],
        "input_dtypes_by_rank": [
            {"rank": item["rank"], "dtypes": [item["input_dtype"]]}
            for item in sorted(participants, key=lambda value: value["rank"])
        ],
    }


def parse_oracle_text(text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    barrier_count = 0
    for message in _messages(text):
        if any(message.startswith(value) for value in BARRIER_TEXTS):
            barrier_count += 1
            continue
        parsed = _parse_collective_message(message)
        if parsed is not None:
            records.append(parsed)

    primary = next(
        (
            divergence
            for record in records
            if (divergence := _primary_divergence(record)) is not None
        ),
        None,
    )
    bounded = {
        "mismatch_detected": bool(records),
        "monitored_barrier_error_count": barrier_count,
        "records": records,
        "primary_divergence": primary,
    }
    canonical = json.dumps(bounded, sort_keys=True, separators=(",", ":"))
    return {
        **bounded,
        "normalized_fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "raw_persisted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    result = parse_oracle_text(args.log.read_text(encoding="utf-8", errors="replace"))
    print(json.dumps(result, indent=2, sort_keys=True))
    has_oracle = (
        result["primary_divergence"] is not None
        or result["monitored_barrier_error_count"]
    )
    return 0 if has_oracle else 1


if __name__ == "__main__":
    raise SystemExit(main())
