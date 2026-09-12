# Organic-hang local gate result

Date: 2026-09-10

Decision: **READY TO RENT FOUR IDENTICAL GPUS FOR INSTANCE SMOKE TESTS**.
Formal A/B/C/D evidence remains prohibited until both first-instance smoke
gates in `EXPERIMENT_PROTOCOL.md` pass.

## Completed locally

- Generated three sanitized fixtures from genuine PyTorch 2.11.0+cpu
  ProcessGroupWrapper/Gloo output: a valid long-form mismatch, a degraded
  fingerprint and a monitored-barrier timeout.
- Corrected the parser for nested `TensorOptions(...)`, the observed
  `failed to pass monitoredBarrier` wording, and degraded records that cannot
  support a unique primary divergence. Degraded evidence now fails closed
  instead of crashing or being promoted to a diagnosis.
- Added four-rank lifecycle helpers with a monotonic wall-clock deadline,
  atomic PID-file binding, `/proc/<pid>/stat` start-time identities, bounded
  TERM/KILL escalation and no-orphan verification.
- Added a real POSIX process-group cleanup test. It is intentionally skipped on
  Windows and will run in Linux CI or the instance smoke phase.
- Confirmed that official Linux CPython 3.12 wheels exist for
  `torch==2.11.0+cu130` and `torch==2.13.0+cu130` in the PyTorch CUDA 13.0
  index. The exact two-venv commands are frozen in `GPU_RUNBOOK.md`.

## Verification

```text
unittest: 61 tests OK, 2 platform skips
Phase 0 verifier: PASS (7 summaries)
Phase 0.5 verifier: PASS
Phase 1 revised verifier: PASS (9 trials)
organic-hang Ruff scope: PASS
compileall: PASS
fixture manifest JSON: PASS
```

The two skips are the existing owner-only POSIX permission test and the new
real POSIX process-group cleanup test. Neither can be exercised faithfully on
Windows.

## First-instance smoke gates

Before reading any formal result:

1. run the real Linux four-rank process tree through the PID/start-time cleanup
   path and confirm no tracked process remains;
2. generate and decode one genuine PyTorch 2.11 Flight Recorder pickle and
   confirm the actual process-group, shape and issuing-thread fields match the
   frozen normalizer contract.

Failure of either gate stops the campaign and produces a setup report, not a
partial scientific result.
