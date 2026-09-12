# Four-GPU organic-hang result

Date: 2026-09-12

Author-declared protocol revision: `2026-09-12.6`

Topology: one host, 4 x NVIDIA GeForce RTX 4090, PP=2, DP=2

Driver: `580.105.08`; CUDA runtime: `13.0`

## Result

The affected-build diagnosis passed, but the campaign cannot claim GO because
it lacks a same-version no-divergence control and the author-declared
cross-version opt-in control did not complete.

| Gate | Trials | Result |
| --- | ---: | --- |
| DETAIL independent oracle on PyTorch 2.11.0 | 3 | PASS; stable `_REDUCE_SCATTER_BASE` shape mismatch |
| Automatic Flight Recorder on PyTorch 2.11.0 | 3 | PASS; same semantic mismatch in 3/3 |
| Optional external-stack arm | 0 | Deferred; it is not required to rescue or establish Gate B |
| Cross-version opt-in control | 3 | BLOCKED; a different mixed-dtype assertion occurs in 3/3 |

The DETAIL and Flight Recorder records use different process-group rank and
sequence-number spaces. The comparison therefore uses operation, group
cardinality and unordered input-shape multisets. DETAIL renders the uniform
input/output dtype pair as `Float Float`, whereas the retained Flight Recorder
summary stores the input dtype as `Float`; the post-run sensitivity verifier
therefore compares their uniform dtype family rather than claiming byte-equal
dtype arrays. It does not force rank labels or sequence counters to match
across the two sources.

The Flight Recorder primary belongs to the expected stage-0 DP group `[0, 2]`.
The normalizer's v1 selection order is deterministic by group and sequence; it
must not be interpreted as proving which divergent group occurred first.

## Cross-version opt-in control finding

`torch==2.12.0+cu130` was rejected as a control because it does not expose
`FSDPModule.set_reduce_scatter_unused_params`. The official PyTorch v2.12.0
source has the same boundary.

`torch==2.13.0+cu130` exposes the API. The prepared control differs from the
affected script only by:

```python
model.set_reduce_scatter_unused_params(True)
```

The prepared control SHA-256 is
`430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1`.
Preflight passed before the trials.

All three trials launched four ranks and then exited in 10-11 seconds with:

```text
AssertionError: FSDP reduce-scatter expects uniform gradient dtype but got
{torch.bfloat16, torch.float32}
```

The workload configures `MixedPrecisionPolicy(param_dtype=torch.bfloat16,
reduce_dtype=torch.float32)`. The opt-in fix supplies zero gradients for unused
parameters, but the zero-gradient path follows the parameter dtype. It can
therefore mix bfloat16 zeros with float32 computed gradients in the same
reduce-scatter input. This is a candidate upstream defect; it is not counted as
a successful control, and no threshold was changed after observing it.

## Safety and retention

- Each affected-build trial and each control attempt recorded four rank PIDs
  during execution.
- The derived public package retains lifecycle JSON for DETAIL trials 2 and 3
  and all three automatic-hang trials; every retained record reports no orphaned
  tracked processes. DETAIL trial 1 has no retained lifecycle JSON, so its
  cleanup result is not independently provable from the public package.
- Raw stderr and prepared scripts remain in the private audit archive. They are
  not part of the public evidence directory.
- The private audit archive is `organic-evidence-private-audit-20260912.tgz`,
  SHA-256
  `24775e7f288b4b85c053e0164a179d3f58d4ed564535982359f2ef159cb8dfc6`.
- The derived-only public evidence is under
  `results/organic-hang-20260912/` and includes a standalone post-run verifier.
- Its packaged copy is `organic-evidence-public-derived-20260912.tgz`, SHA-256
  `06917a2e0872e2535dd842ba01debd6b10d12621d7dfd6b24c0b8374fb1a75d2`.

The experiment directory was untracked at execution time. Git history therefore
cannot independently prove that protocol revision `2026-09-12.6` preceded the
formal trials. The rules are preserved as an author-declared record, not a
cryptographically verifiable pre-run freeze. Future campaigns must embed a
committed protocol SHA in every result before execution.

## Interpretation

This campaign establishes that the retained automatic Flight Recorder evidence
reconstructs the known operation and input-shape mismatch on an organic
workload. A post-run sensitivity check additionally confirms the expected
stage-0 DP group and uniform Float dtype family.
It does not establish that the upstream fix resolves the original workload.
The next step is to reduce the mixed-dtype control failure to a small upstream
reproducer and ask PyTorch maintainers whether zero gradients should be created
at the configured reduce dtype. After an accepted fix or a pinned build with
that behavior exists, rerun the three fixed-control trials unchanged.
