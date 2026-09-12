# Pre-registered protocol: organic FSDP2 collective divergence

Protocol revision: `2026-09-12.6`

## Post-run audit note

This directory was untracked when the formal trials ran. Git history cannot
independently verify the freeze time of this author-declared protocol. The raw
audit archive also shows that DETAIL combines input/output dtype labels as
`Float Float`, while the retained Flight Recorder primary stores input dtype as
`Float`; the old wording below that requires exact dtype-multiset equality is
superseded by the published post-run dtype-family sensitivity check. Arm D is a
cross-version PyTorch 2.13 opt-in control, not proof of default fixed behavior.
Its execution exposed a different mixed-gradient-dtype assertion in 3/3 trials.

These corrections do not alter the historical decision rule. The next campaign
rules are specified prospectively in `NEXT_CAMPAIGN_PLAN.md` and must be
committed before execution.

## Question

Can bounded Flight Recorder evidence from the default hung run identify the
divergent collective and agree with PyTorch's independent DETAIL-mode oracle?
Separately, does rank-associated external stack evidence change an actionable
output after Flight Recorder has already identified a same-sequence mismatch?

## Non-claims

This experiment does not claim to discover the already-published root cause,
prove production coverage, validate vLLM, or establish cross-host clock order.
It tests diagnostic reconstruction on an upstream failure not designed for the
lab.

## Frozen source and topology

- Reproducer and hash: `SOURCE_AUDIT.md`
- Affected build: exact `torch==2.11.0` variant selected for the rental image
- Fixed control: exact `torch==2.13.0` compatible variant
- World size: 4
- Pipeline parallel size: 2
- FSDP data-parallel shard size: 2
- Seed: 0 plus global rank, unchanged from upstream
- Model, branch probability, optimizer, batch size, and 200-step target:
  unchanged from upstream
- Process-group timeout: 60 seconds, explicitly expressed as seconds and
  applied to the default group and every DeviceMesh-created child group
- Autograd anomaly detection: disabled in every arm
- DETAIL and Flight Recorder sequence numbers are retained separately. They
  belong to different instrumentation layers and are not used as a join key.

Any topology or fault-mechanism change requires a new protocol revision before
results are read.

The child-group clause and the semantic oracle comparison below were frozen
before valid Arm B evidence was collected.
Setup attempt B1 on 2026-09-10 confirmed that PyTorch 2.11 otherwise retains
the 10-minute NCCL default for DeviceMesh-created groups; it produced no
automatic artifact within the 150-second external deadline and is retained as
a setup exclusion, not scored as a formal trial.

## Arms and gates

### A. DETAIL oracle on affected build

Run three trials with `DFX_ORGANIC_DEBUG_DETAIL=1`. Autograd anomaly detection
remains disabled, so DETAIL wrapping is the only diagnostic variable changed.
This is a declared deviation from the reporter's `debug=True` path, which also
enabled anomaly detection. Do not attach py-spy or score Flight Recorder.
Capture a bounded, redacted error summary.

Gate A passes only if all three trials emit the same typed mismatch for
`_REDUCE_SCATTER_BASE`, with a sequence number, process-group participants and
input-buffer shape disagreement. FSDP2 calls `reduce_scatter_tensor`, which is
recorded by DETAIL as `_REDUCE_SCATTER_BASE`; DETAIL prints the input buffer's
shape for this path. Save one structured record per mismatch message and a
normalized primary divergence, not raw stderr. Stop the campaign if no such
mismatch appears within the unchanged 200-step target.

### B. Automatic timeout on affected build

Set `DFX_ORGANIC_DEBUG_DETAIL=0`; enable the production automatic Flight
Recorder timeout path without a debug pipe. Do not attach py-spy before the
automatic artifacts exist.

Run three trials. Gate B passes only if all trials:

1. enter the reported hang rather than a setup/import failure;
2. produce four bounded per-rank Flight Recorder artifacts or an explicitly
   typed miss;
3. yield one stable normalized primary divergence across trials;
4. agree with Gate A on operation, subgroup cardinality, and the unordered
   input-shape and dtype multiset; and
5. clean all four ranks without orphans.

A typed miss is reported but does not satisfy diagnostic validity.

Flight Recorder normalization is process-group aware. Its key is sorted global
rank membership from `pg_config` plus the collective sequence number. P2P
SEND/RECV records and COALESCED wrapper records cannot define a comparable
cross-rank collective signature and are excluded from primary selection.
Per-rank values retain operation, input/output shape and dtype, state,
`record_id`, `thread_id`, and `thread_name`. Entries are ordered only within one
rank by `record_id`; wall-clock timestamps never order ranks. Duplicate
comparable entries for one rank/key are a typed ambiguity rather than silently
collapsed. The primary divergence is the first complete collective key whose
members disagree on operation, shape, or dtype. Missing-member, all-pending and
ambiguous keys are secondary effects and are never compared to the oracle.
Every primary-group member must have retained exactly one primary entry, and at
least one earlier completed collective must prove that the run progressed
beyond setup.

DETAIL reports ranks in the wrapped process group's local rank space, while
Flight Recorder retains global membership through `pg_config`. Their sequence
counters also originate in different instrumentation layers. Therefore Gate B
does not equate rank labels or sequence values across the two sources. It
requires the same operation, subgroup cardinality, and unordered input-shape
and dtype multiset. Both sources' rank labels and sequence numbers remain in
the retained summaries so this boundary is auditable rather than hidden.

### C. Post-gate external stack sampling

Arm C is a separate three-trial run, not sampling of Arm B. DETAIL stays off.
Before the automatic 60-second timeout tears down ranks, request a manual Flight
Recorder dump and then collect three spaced native py-spy samples per rank.
Sampling is suspending and is not permitted on healthy progress.

External stacks receive incremental-value credit only if they change a
pre-registered output: divergent-rank interpretation, first actionable CPU
location, or next diagnostic action. More detail without changing one of those
outputs is recorded as corroboration, not incremental value.

Flight Recorder `thread_id` is joined to py-spy's `os_thread_id`; the observed
`/proc/<pid>/task/<tid>/comm` is retained as corroboration. This handles the
`pt_autograd_<device>` worker that issues FSDP2 backward collectives instead of
assuming the main thread. Failure to sample that issuing thread is a typed
outcome.

Frame provenance is frozen: a project frame must resolve to the exact prepared
target path. Function names do not select it; only basename, function and line
are retained. Provenance chooses what to report but earns no value by itself.
The pre-registered expected result is **no incremental value**: Flight Recorder
should already identify this same-sequence size mismatch. Credit is possible
only if stacks change divergent-rank interpretation, first actionable CPU
location, or next diagnostic action.

### D. Fixed-version control

Run three trials of the same prepared reproducer on the exact PyTorch 2.13.0
control build with DETAIL disabled. The control passes only if all ranks
complete all 200 configured training steps, exit normally, and leave no orphan.
This establishes that the
observed affected-build signature is absent after the pinned upstream fix; it
does not prove all conditional-computation cases are fixed.

Arm D uses the same prepared workload with only
`model.set_reduce_scatter_unused_params(True)` added. Its prepared SHA-256 is
`430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1`;
the schema and preflight reject this hash in A/B/C and reject the affected-build
hash in D.

## Decision rule

- **GO:** A supplies a stable independent oracle in 3/3; B reproduces a stable
  matching diagnosis in 3/3 without external sampling; D completes in 3/3.
- **CONDITIONAL GO:** B yields useful but unstable or partial evidence, with
  every miss published and no post-hoc threshold change.
- **NO-GO:** the affected build does not reproduce under the frozen topology,
  B disagrees with the independent oracle, or the fixed control retains the
  same failure signature.

Arm C is not required for GO. Its purpose is to measure the boundary between
Flight Recorder and external stack value, not to rescue a failed Gate B.

## Safety and retention

- Hard wall-clock deadline per arm and out-of-band cleanup are mandatory.
- Raw stderr, py-spy JSON, and Flight Recorder pickle files are transient.
- Retained output contains allow-listed fields, normalized fingerprints,
  monotonic offsets, tool/runtime/driver versions, embedded preflight results,
  trigger provenance, the effective 200-step target, and cleanup status.
- No prompt, token, model weight, environment dump, absolute user path, host
  identifier, or raw command line is retained.
- Any observer-attach failure is typed and fail-open; it cannot erase Flight
  Recorder evidence or block cleanup.

## Pre-rental exit checklist

- [x] Preparation script passes against the pinned source hash.
- [x] Prepared source compiles and a diff contains only declared hooks.
- [x] Result schema and verifier are frozen.
- [x] Four-rank lifecycle helpers have a wall-clock deadline, bounded TERM/KILL
      cleanup and `/proc` PID start-time reuse protection. The Linux GPU run
      must still exercise the real process tree before formal evidence is read.
- [x] DETAIL parser passes genuine CPU/Gloo long-form, degraded-fingerprint and
      monitored-barrier fixtures; no hand-written fixture is used as evidence.
- [x] Flight Recorder parser has synthetic four-rank, multiple-process-group,
      reduce-scatter shape and issuing-thread identity coverage.
- [x] Arm C manual trigger, three-trial count, thread join and exact-path frame
      provenance rule are frozen.
- [x] Official Linux CPython 3.12 CUDA 13.0 wheels exist for exact affected
      `torch==2.11.0+cu130` and fixed `torch==2.13.0+cu130` environments. The
      per-instance preflight must still record and match the runtime, NCCL,
      driver and GPU identities before formal trials.

## First-instance smoke gates before formal trials

The instance may be rented after the checklist above passes, but no A/B/C/D
result becomes formal until both smoke gates pass:

- [ ] Decode one genuine PyTorch 2.11 Flight Recorder pickle and confirm the
      observed process-group, shape and issuing-thread fields match the frozen
      normalizer input contract.
- [ ] Exercise the real Linux four-rank process tree through the bounded
      deadline and PID/start-time cleanup path, then confirm no tracked process
      remains.
