from __future__ import annotations

import math
import re
from dataclasses import dataclass


SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]Inf)"
    r"(?:\s+\d+)?$"
)

DEFAULT_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
}


@dataclass(frozen=True, slots=True)
class MetricSample:
    name: str
    value: float


def parse_samples(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        value = float(match.group("value"))
        if math.isfinite(value):
            samples.append(MetricSample(match.group("name"), value))
    return samples


def select_metrics(
    text: str, names: set[str] | None = None
) -> dict[str, float]:
    """Aggregate matching label variants without creating high-cardinality output."""
    selected = names or DEFAULT_METRICS
    result: dict[str, float] = {}
    for sample in parse_samples(text):
        if sample.name in selected:
            result[sample.name] = result.get(sample.name, 0.0) + sample.value
    return result

