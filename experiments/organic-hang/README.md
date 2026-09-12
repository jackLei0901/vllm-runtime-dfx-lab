# Organic distributed-hang validation

This directory prepares the first non-author-designed validation case for the
runtime reliability lab. The selected case is the FSDP2 conditional-parameter
hang reported in PyTorch issue #158719 and reposted as TorchTitan issue #2747.

The case is not evidence of discovering a new root cause. Its purpose is to
test whether the lab reconstructs a known collective mismatch from a hung run
and agrees with PyTorch's independent `DebugLevel.DETAIL` oracle.

## Current result

The affected-build reconstruction passed: three DETAIL trials produced a
stable oracle and three automatic Flight Recorder trials reproduced the same
operation and input-shape mismatch. The campaign still lacks a same-version
no-divergence control. Its cross-version opt-in control is also blocked, not
passed: PyTorch 2.13.0 exposes the API, but the original mixed-precision
workload fails 3/3 with a uniform-gradient-dtype assertion. See
[`GPU_RESULT_2026-09-12.md`](GPU_RESULT_2026-09-12.md).

## Why this case

- It is an organic upstream failure rather than a lab injection.
- FSDP2 collectives pass through c10d/ProcessGroupNCCL, so Flight Recorder can
  observe the relevant distributed state.
- The reporter's `DETAIL` run changes the hang into a typed collective-mismatch
  error, providing an independent oracle.
- The reproducer is small but requires four GPUs (`PP=2`, `DP=2`).
- PyTorch 2.11.0 and 2.12.0 do not expose the fix API; 2.13.0 contains it.

## Source handling

The upstream Gist does not declare a license. This repository therefore does
not redistribute it. `fetch_and_prepare_reproducer.py` downloads or accepts a
local copy of one pinned revision, requires its SHA-256 to match, and applies
only reviewed environment/observer hooks. A changed upstream file fails
closed.

```bash
python experiments/organic-hang/fetch_and_prepare_reproducer.py \
  --output /tmp/pp_fsdp_graph_test.prepared.py
```

The transformation does not change the model, branch condition, seed,
parallel dimensions, batch size, optimizer, or gradient behavior. It changes:

- `timedelta(60)` from 60 days to an explicit configurable number of seconds;
- the same bounded timeout applied to every DeviceMesh-created child process
  group, which otherwise retains PyTorch 2.11's 10-minute NCCL default;
- the hard-coded rendezvous port to an environment-controlled port;
- DETAIL mode and training-step count to environment-controlled values;
- autograd anomaly detection to disabled in every arm, so DETAIL is the only
  oracle variable;
- scoped `PR_SET_PTRACER` authorization for the declared observer; and
- one per-rank PID file after process-group initialization.

The result contract stores a structured `primary_divergence` on both the DETAIL
and Flight Recorder sides. Gate B compares operation, subgroup cardinality and
the unordered input-shape multiset. DETAIL's group-local rank labels and
sequence counter are retained separately from Flight Recorder's global group
identity and counter; they are not incorrectly treated as a shared join key. A
post-run sensitivity check also verifies the expected stage-0 DP group and
uniform dtype family. Arm C uses separate manual-dump trials and is expected to
add no value for this collective mismatch.

Published derived evidence and its standalone verifier are under
[`results/organic-hang-20260912/`](../../results/organic-hang-20260912/). The raw
toy-workload archive is retained privately and must not be published. Because
this experiment directory was untracked during execution, the protocol's
pre-run freeze timing is author-declared rather than independently Git-verifiable.

The current English review entry is
[`REVIEW_RESPONSE_2026-09-10.md`](REVIEW_RESPONSE_2026-09-10.md); the Chinese
entry remains [`REVIEW_START_HERE_CN.md`](REVIEW_START_HERE_CN.md). Read
`SOURCE_AUDIT.md` and `EXPERIMENT_PROTOCOL.md` before interpreting the result.
The exact environment pair, preflight commands, arm order and cleanup contract
are collected in [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md).
