from __future__ import annotations

import argparse
import json
from pathlib import Path

from dfxlab.collectors import environment_snapshot
from dfxlab.faults import inject_signal
from dfxlab.recorder import IncidentRecorder
from dfxlab.report import summarize_file
from dfxlab.schema import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-dfx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("snapshot-env", help="capture environment")
    env_parser.add_argument("--output", type=Path)

    record_parser = subparsers.add_parser("record", help="record bounded runtime history")
    record_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    record_parser.add_argument("--pid", type=int)
    record_parser.add_argument("--output", type=Path, required=True)
    record_parser.add_argument("--interval", type=float, default=1.0)
    record_parser.add_argument("--duration", type=float)
    record_parser.add_argument("--history", type=int, default=300)
    record_parser.add_argument("--timeout", type=float, default=1.0)
    record_parser.add_argument("--kv-threshold", type=float, default=0.95)
    record_parser.add_argument("--unhealthy-samples", type=int, default=3)
    record_parser.add_argument("--preemption-delta", type=float, default=20.0)
    record_parser.add_argument("--incident-cooldown", type=float, default=60.0)
    record_parser.add_argument("--stop-on-incident", action="store_true")

    signal_parser = subparsers.add_parser("inject-signal", help="send an explicit signal")
    signal_parser.add_argument("--pid", type=int, required=True)
    signal_parser.add_argument("--signal", default="SIGKILL")
    signal_parser.add_argument("--event-log", type=Path)
    signal_parser.add_argument("--dry-run", action="store_true")

    summary_parser = subparsers.add_parser("summarize", help="render incident markdown")
    summary_parser.add_argument("--input", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot-env":
        payload = environment_snapshot()
        if args.output:
            atomic_write_json(args.output, payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "record":
        recorder = IncidentRecorder(
            base_url=args.base_url,
            output_dir=args.output,
            pid=args.pid,
            interval=args.interval,
            history_size=args.history,
            timeout=args.timeout,
            kv_threshold=args.kv_threshold,
            unhealthy_samples=args.unhealthy_samples,
            preemption_delta=args.preemption_delta,
            incident_cooldown=args.incident_cooldown,
        )
        return recorder.run(args.duration, args.stop_on_incident)
    if args.command == "inject-signal":
        event = inject_signal(
            args.pid, args.signal, args.event_log, dry_run=args.dry_run
        )
        print(json.dumps(event, ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize":
        summarize_file(args.input, args.output)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
