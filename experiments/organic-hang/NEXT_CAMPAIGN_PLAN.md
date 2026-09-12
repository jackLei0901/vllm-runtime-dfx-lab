# Post-run amendments for the next organic-hang campaign

Status: **design complete; GPU execution not started.**

This document was written after the 2026-09-12 campaign. It does not alter the
historical result or retroactively claim pre-registration.

## Required same-version negative control

The next campaign adds a no-divergence arm on the affected PyTorch 2.11.0
environment. It preserves the source, model, seed, `PP=2`/`DP=2` topology,
mixed-precision policy, training steps and timeout. The only workload change is:

```python
enable_random_output=False
```

Generate it with:

```bash
python fetch_and_prepare_reproducer.py \
  --disable-random-output \
  --output /tmp/pp_fsdp_graph_test.no-divergence.py
```

Expected prepared-source SHA-256:
`bb4126cd3a74a57d1b58b11035876753871e599076728c559c3a5f3d46fa0201`.

Run three independent trials. Each trial must:

- launch four ranks on PyTorch 2.11.0;
- complete all 200 training steps and exit zero;
- produce no DETAIL collective mismatch and no Flight Recorder primary
  divergence;
- retain four rank PID identities and report no tracked orphan; and
- embed the committed protocol SHA and prepared-source SHA in its summary.

Any primary divergence or incomplete training run fails the negative-control
gate. It must not be reclassified after inspecting the output.

## Verifier amendments

- The case-specific Flight Recorder primary must have global group membership
  `[0, 2]`, the stage-0 DP group where the conditional parameters live.
- DETAIL's `Float Float` combined input/output representation and Flight
  Recorder's split `Float` input representation are compared as one uniform
  dtype family. Exact array equality is not claimed.
- Normalizer v1's lexicographic group/sequence selection is deterministic, not
  temporal. No result may call it the "first" divergent group.
- A future temporal-primary claim requires a separately designed logical-order
  contract; rank-local `record_id` and unsynchronized wall clocks are
  insufficient.

## Freeze procedure

Before GPU execution:

1. commit the protocol, preparer, schema, verifier and tests;
2. record that commit SHA in the run manifest and every result summary;
3. generate all prepared sources and verify their exact hashes;
4. run the CPU tests and four-GPU preflight; and
5. make no acceptance-rule changes until every trial is published.

The current repository commit cannot prove the freeze date of the 2026-09-12
campaign. This procedure applies prospectively.
