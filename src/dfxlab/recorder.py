from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from dfxlab.collectors import collect_sample, environment_snapshot
from dfxlab.schema import Incident, Sample, atomic_write_json, utc_now


KV_METRICS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
PREEMPTION_METRIC = "vllm:num_preemptions_total"


class IncidentRecorder:
    def __init__(
        self,
        base_url: str,
        output_dir: Path,
        pid: int | None = None,
        interval: float = 1.0,
        history_size: int = 300,
        timeout: float = 1.0,
        kv_threshold: float = 0.95,
        unhealthy_samples: int = 3,
        preemption_delta: float = 20.0,
        incident_cooldown: float = 60.0,
    ) -> None:
        if interval <= 0 or history_size <= 0:
            raise ValueError("interval and history_size must be positive")
        self.base_url = base_url
        self.output_dir = output_dir
        self.pid = pid
        self.interval = interval
        self.timeout = timeout
        self.kv_threshold = kv_threshold
        self.unhealthy_samples = unhealthy_samples
        self.preemption_delta = preemption_delta
        self.incident_cooldown = incident_cooldown
        self.history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.environment = environment_snapshot()
        self._healthy_once = False
        self._consecutive_unhealthy = 0
        self._last_preemptions: float | None = None
        self._last_capture_by_reason: dict[str, float] = {}

    def classify(self, sample: Sample) -> tuple[str, str] | None:
        process = sample.process
        if process and not process.get("alive", True):
            return "process_exit", "fatal"

        if sample.health.get("ok"):
            self._healthy_once = True
            self._consecutive_unhealthy = 0
        elif self._healthy_once:
            self._consecutive_unhealthy += 1
            if self._consecutive_unhealthy >= self.unhealthy_samples:
                return "health_lost", "fatal"

        for name in KV_METRICS:
            usage = sample.metrics.get(name)
            if usage is not None and usage >= self.kv_threshold:
                return "kv_pressure", "warning"

        current = sample.metrics.get(PREEMPTION_METRIC)
        if current is not None:
            if (
                self._last_preemptions is not None
                and current - self._last_preemptions >= self.preemption_delta
            ):
                self._last_preemptions = current
                return "preemption_storm", "warning"
            self._last_preemptions = current
        return None

    def capture(self, reason: str, severity: str) -> Path:
        stamp = utc_now().replace(":", "-")
        path = self.output_dir / f"incident-{stamp}.json"
        incident = Incident(
            reason=reason,
            severity=severity,
            captured_at=utc_now(),
            environment=self.environment,
            samples=list(self.history),
            metadata={
                "base_url": self.base_url,
                "pid": self.pid,
                "history_capacity": self.history.maxlen,
            },
        )
        atomic_write_json(path, incident.to_dict())
        return path

    def run(self, duration: float | None = None, stop_on_incident: bool = False) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = self.output_dir / "timeline.jsonl"
        start = time.monotonic()
        total_samples = 0
        captured: list[str] = []
        with timeline_path.open("a", encoding="utf-8") as timeline:
            try:
                while duration is None or time.monotonic() - start < duration:
                    sample = collect_sample(self.base_url, self.pid, self.timeout)
                    payload = sample.to_dict()
                    self.history.append(payload)
                    total_samples += 1
                    timeline.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    timeline.flush()
                    incident = self.classify(sample)
                    if incident:
                        reason, severity = incident
                        now = time.monotonic()
                        last_capture = self._last_capture_by_reason.get(reason)
                        if (
                            last_capture is None
                            or now - last_capture >= self.incident_cooldown
                        ):
                            path = self.capture(reason, severity)
                            self._last_capture_by_reason[reason] = now
                            captured.append(str(path))
                            print(f"captured {severity} incident: {path}")
                            if stop_on_incident:
                                break
                    time.sleep(self.interval)
            except KeyboardInterrupt:
                print("recording stopped by user")
        atomic_write_json(
            self.output_dir / "run-summary.json",
            {
                "schema_version": 1,
                "started_monotonic_seconds": start,
                "finished_at": utc_now(),
                "samples_collected": total_samples,
                "samples_retained": len(self.history),
                "incident_files": captured,
                "timeline": str(timeline_path),
            },
        )
        return 0
