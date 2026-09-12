# Four-GPU organic-hang validation: review entry

Status: **the affected-build reconstruction passed; the campaign-level GO
decision remains blocked by a missing negative control and a new cross-version
opt-in control failure.**

This is the review entry for the first lab case that was not designed by the
project. The selected workload is the FSDP2 conditional-parameter hang reported
in PyTorch issue #158719 and reposted as TorchTitan issue #2747. Its root cause
is already public, so this case tests reconstruction against an independent
answer; it does not demonstrate discovery of an unknown bug.

## What the experiment asked

Can the lab recover the same collective mismatch from a normal hung run that
PyTorch reports when `DebugLevel.DETAIL` is enabled?

The frozen topology is one host with four RTX 4090 GPUs, `PP=2`, `DP=2`. The
affected environment uses PyTorch 2.11.0. The fixed-control environment uses
PyTorch 2.13.0 because 2.12.0 does not expose
`FSDPModule.set_reduce_scatter_unused_params`.

## Result at a glance

| Gate | Trials | Result |
| --- | ---: | --- |
| A: independent DETAIL oracle, PyTorch 2.11.0 | 3 | **PASS**; stable `_REDUCE_SCATTER_BASE` input-shape mismatch |
| B: automatic Flight Recorder, PyTorch 2.11.0 | 3 | **PASS**; the same semantic mismatch in 3/3 |
| C: external native-stack sampling | 0 | Deferred; optional and not permitted to rescue Gate B |
| D: cross-version opt-in control, PyTorch 2.13.0 | 3 | **BLOCKED**; a different mixed-gradient-dtype assertion occurred in 3/3 |

Gate B does not compare unstable labels or assume synchronized clocks. DETAIL
and Flight Recorder use different process-group rank and sequence-number
spaces. The verifier instead compares the collective operation, group
cardinality and unordered input-shape multiset while retaining both identity
spaces for audit. The retained DETAIL summary represents a uniform input/output
dtype pair as `Float Float`; Flight Recorder stores the input dtype as `Float`.
The post-run public verifier compares the uniform dtype family rather than
claiming exact equality between those differently shaped fields.

The selected Flight Recorder primary belongs to the expected stage-0 DP group
`[0, 2]`. Normalizer v1 selects deterministically by group and sequence; its
`primary` label is not a claim that this was the temporally first divergent
group.

## The cross-version opt-in control blocker

The fixed-control source differs from the affected prepared workload only by:

```python
model.set_reduce_scatter_unused_params(True)
```

Its prepared-source SHA-256 is
`430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1`,
and the four-GPU preflight passed before execution.

All three control trials launched four ranks, then exited in 10-11 seconds with:

```text
AssertionError: FSDP reduce-scatter expects uniform gradient dtype but got
{torch.bfloat16, torch.float32}
```

The workload uses
`MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`.
The opt-in unused-parameter path creates zero gradients following the parameter
dtype, while computed gradients may follow the configured reduce dtype. The
result can mix bfloat16 zeros and float32 gradients in one reduce-scatter input.

This is recorded as a candidate upstream defect, not as a successful fixed
control. No threshold, oracle, or GO rule was changed after observing it. The
next step is a minimal PyTorch reproducer and maintainer confirmation of the
expected dtype contract; only then should the three D trials be rerun.

## Claims this evidence supports

- On this organic workload, the lab reconstructed the independently known
  collective mismatch from automatic Flight Recorder evidence in 3/3 trials.
- The semantic comparison remained stable despite different local/global rank
  labels and sequence counters.
- All retained lifecycle records show bounded cleanup with no tracked orphan
  processes. DETAIL trial 1 has no lifecycle JSON in the retained package.
- The fixed-control attempt surfaced a separate, repeatable compatibility
  problem rather than silently being counted as evidence for success.

## Claims it does not support

- It does not show that the upstream opt-in fix resolves the original workload.
- It does not demonstrate root-cause discovery; the answer was already known.
- It does not establish incremental value from external stack sampling because
  Arm C was not run.
- It does not establish cross-host correlation, vLLM hot-path coverage, or a
  generally applicable hang diagnosis rate.
- It does not show that the pipeline stays silent on a same-version workload
  with no collective divergence; that negative control remains required.
- It does not justify changing the campaign decision to GO.

## Suggested review order

1. [`GPU_RESULT_2026-09-12.md`](GPU_RESULT_2026-09-12.md): delivered results,
   limitations, retention boundary, and evidence hash.
2. [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md): upstream provenance, version boundary,
   original hash, and prepared-source hashes.
3. [`fetch_and_prepare_reproducer.py`](fetch_and_prepare_reproducer.py): confirm
   that the affected workload is unchanged apart from declared controls and
   observer hooks, and that the fixed arm adds only the opt-in API call.
4. [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md): inspect the four arms,
   three-trial rule, semantic oracle, and author-declared decision rules. Git
   history cannot independently prove their pre-run freeze date.
5. [`organic-hang-result-v1.schema.json`](organic-hang-result-v1.schema.json):
   confirm that affected and fixed arms require the correct PyTorch version and
   exact prepared-source hash.
6. [`parse_detail_oracle.py`](parse_detail_oracle.py) and
   [`normalize_flight_recorder.py`](normalize_flight_recorder.py): inspect the
   allow-listed transformations from the two evidence sources.
7. [`verify_organic_results.py`](verify_organic_results.py): inspect whether a
   missing trial, wrong semantic mismatch, cross-arm source hash, unstable
   signature, or failed fixed control could be accepted.
8. [`preflight.py`](preflight.py), [`process_lifecycle.py`](process_lifecycle.py),
   and [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md): inspect environment identity,
   deadline enforcement, PID reuse protection, teardown, and retention.

## Verification performed

- Remote four-GPU environment: 30 organic-hang tests passed.
- Ruff rule check and Ruff format check passed for the experiment and its tests.
- Local recheck: 30 tests passed; schema-dependent cases were skipped where the
  optional `jsonschema` package was unavailable.
- `git diff --check` passed.
- No `torchrun`, reproducer, sampler, or tracked GPU process remained after the
  campaign.
- The private audit archive is `organic-evidence-private-audit-20260912.tgz`,
  SHA-256
  `24775e7f288b4b85c053e0164a179d3f58d4ed564535982359f2ef159cb8dfc6`.

The derived-only public package is under `results/organic-hang-20260912/` and
contains a standalone post-run verifier. Its archive is
`organic-evidence-public-derived-20260912.tgz`, SHA-256
`06917a2e0872e2535dd842ba01debd6b10d12621d7dfd6b24c0b8374fb1a75d2`.
Raw stderr and prepared scripts remain only in the private audit archive; the
private archive must not be published.

The experiment directory was untracked at execution time. Protocol revision
`2026-09-12.6` is preserved as the author's declared pre-run record, but its
freeze timing is not independently verifiable from Git history. Future campaigns
must embed a committed protocol SHA in every summary before execution.

## Questions for the reviewer

1. Is the A/B semantic oracle strict enough to prevent a stable but incorrect
   diagnosis from passing?
2. Does the fixed-control failure invalidate only Gate D, or expose a confound
   in the affected-build A/B reconstruction?
3. Is the mixed-gradient-dtype explanation sufficiently grounded to justify a
   separate minimal PyTorch issue, without yet claiming a root cause?
4. Are the stated non-claims and public/private evidence boundary adequate?

The next frozen campaign must add a same-version no-divergence control before
the four-GPU run. It keeps PyTorch 2.11.0, topology and source constant, changes
only every layer's `enable_random_output` to `False`, and requires 200 completed
steps with no primary divergence.

## Product milestone after this case

This known-answer reconstruction is maturity step 2, not the final product
claim. The next milestone remains:

> Evidence produced by the lab helped diagnose a real issue that was still open
> and unresolved, and caused an observable change in its public investigation.

Diagnosis and resolution remain separate achievements. Candidate issues must
be selected before reading an answer, and misses or unanswered contributions
must be published alongside successful attempts.
