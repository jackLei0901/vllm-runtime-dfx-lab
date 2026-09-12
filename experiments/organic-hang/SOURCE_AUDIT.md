# Source and fix audit: PyTorch #158719 / TorchTitan #2747

Audit date: 2026-09-10 UTC

## Canonical records

- PyTorch issue
  [`pytorch/pytorch#158719`](https://github.com/pytorch/pytorch/issues/158719),
  opened 2025-07-20 and closed 2026-04-04.
- Open TorchTitan repost
  [`pytorch/torchtitan#2747`](https://github.com/pytorch/torchtitan/issues/2747),
  opened 2026-03-30.
- Reporter reproducer Gist
  [`170c4fdd44ab731441910e542d71b24e`](https://gist.github.com/man2machine/170c4fdd44ab731441910e542d71b24e).
- Fix commit
  [`775500a5991db0967e96cb7d1ebf21efe055c9dc`](https://github.com/pytorch/pytorch/commit/775500a5991db0967e96cb7d1ebf21efe055c9dc),
  which names PyTorch PR #170667 and `Fixes #158719`.

GitHub reports PR #170667 as closed and not merged, while the generated fix
commit is present in the PyTorch repository and closed #158719. The experiment
therefore pins the commit, not the PR's merge-status field.

## Pinned reproducer

- File: `pp_fsdp_graph_test.py`
- Raw revision URL: encoded in `fetch_and_prepare_reproducer.py`
- UTF-8 bytes: 24,863
- SHA-256:
  `f23a41bebca5bcac51c6433ecc4a837fa3bbd1b5fd552c6701373619fafe0654`
- Deterministically prepared UTF-8 bytes: 26,015
- Prepared SHA-256:
  `47fce815563b5ce8e94f93e18f1184b73fa0382ccd3934b7df02a90590c0a5a2`

The source uses 32 layers of width 7 and reports that the reduced reproducer is
finicky: it reproduces with four processes at `PP=2`, `DP=2`, but not with the
provided `PP=1`, `DP=4` arrangement. The branch is seeded per global rank and
may use different parameter sets across ranks, producing different gradient
sets and reduce-scatter tensor shapes.

The canonical prepared-source identity is SHA-256
`47fce815563b5ce8e94f93e18f1184b73fa0382ccd3934b7df02a90590c0a5a2`.
The preparer verifies both the input and output identities so newline handling
or an accidental transformation change cannot alter the GPU target silently.

## Independent oracle

With `dist.set_debug_level(dist.DebugLevel.DETAIL)`, the reporter observes a
typed error stating that ranks run mismatched collectives and identifies tensor
shape differences. Without DETAIL, the reported symptom is a hang. The DETAIL
arm is ground truth only; it cannot be counted as evidence from the hung arm.
The upstream `debug=True` path also enables autograd anomaly detection, which
adds per-node host synchronization. The prepared target deliberately disables
anomaly detection in every arm and toggles only DETAIL; this is a declared
experimental deviation from the reporter's diagnostic run.

## Version boundary

GitHub commit ancestry checks against the fix commit produced:

| Tag | Relationship to fix commit | Use |
| --- | --- | --- |
| `v2.11.0` | diverged; does not contain the fix | affected/hang and DETAIL arms |
| `v2.12.0` | release source does not expose the fix API | excluded setup candidate |
| `v2.13.0` | contains the fix and public opt-in API | cross-version opt-in control |

The GPU run must record the exact wheel version, CUDA runtime, NCCL version,
driver, GPU model, source hash, and prepared-source hash. A different PyTorch
version cannot inherit these expectations.

## Post-run control finding

The PyTorch 2.13 control enabled
`model.set_reduce_scatter_unused_params(True)` and passed preflight, but all
three trials stopped before the original divergence with a uniform-gradient-
dtype assertion involving bfloat16 and float32. It is therefore not evidence
that the original workload is fixed. The control is described as cross-version
and opt-in because neither the PyTorch version nor the default behavior matches
the affected arm.

The next same-version no-divergence control changes only
`enable_random_output=(i < (num_stages - 1))` to
`enable_random_output=False`. Its prepared SHA-256 is
`bb4126cd3a74a57d1b58b11035876753871e599076728c559c3a5f3d46fa0201`.
