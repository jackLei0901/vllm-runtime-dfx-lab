from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SENSITIVE_KEYS = {
    "prompt",
    "prompt_token_ids",
    "input",
    "messages",
    "text",
    "json",
    "regex",
    "grammar",
    "structural_tag",
    "authorization",
    "api_key",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def anonymize_id(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def redact(value: Any, key: str | None = None) -> Any:
    """Remove common request content while preserving diagnostic structure."""
    if key and key.lower() in SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return [redact(v) for v in value]
    return value


@dataclass(slots=True)
class Sample:
    timestamp: str
    monotonic_seconds: float
    process: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    gpus: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return redact(asdict(self))


@dataclass(slots=True)
class Incident:
    reason: str
    severity: str
    captured_at: str
    environment: dict[str, Any]
    samples: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return redact(asdict(self))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(redact(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

