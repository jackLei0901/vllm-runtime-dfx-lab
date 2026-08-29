from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def _metric_summary(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for sample in samples:
        for name, value in sample.get("metrics", {}).items():
            values.setdefault(name, []).append(float(value))
    return {
        name: {"min": min(items), "max": max(items), "mean": mean(items)}
        for name, items in sorted(values.items())
    }


def render_incident_markdown(payload: dict[str, Any]) -> str:
    samples = payload.get("samples", [])
    health_failures = sum(not s.get("health", {}).get("ok", False) for s in samples)
    lines = [
        "# Runtime incident summary",
        "",
        f"- Reason: `{payload.get('reason', 'unknown')}`",
        f"- Severity: `{payload.get('severity', 'unknown')}`",
        f"- Captured at: `{payload.get('captured_at', 'unknown')}`",
        f"- Samples retained: `{len(samples)}`",
        f"- Unhealthy samples: `{health_failures}`",
        "",
        "## Metric range",
        "",
        "| Metric | Min | Mean | Max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, stats in _metric_summary(samples).items():
        lines.append(
            f"| `{name}` | {stats['min']:.6g} | {stats['mean']:.6g} | {stats['max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This report describes observed state. It does not infer the root cause without logs, configuration, and a matched control run.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_file(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_incident_markdown(payload), encoding="utf-8")

