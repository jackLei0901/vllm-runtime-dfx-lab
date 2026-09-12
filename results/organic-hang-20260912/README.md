# Published derived evidence: organic FSDP2 hang

This directory contains the allow-listed evidence retained from the four-GPU
campaign run on 2026-09-12. It deliberately excludes raw stderr, stdout,
Flight Recorder pickle files, rank PID files, and prepared source files.

The private audit archive is retained outside version control as
`organic-evidence-private-audit-20260912.tgz`, SHA-256
`24775e7f288b4b85c053e0164a179d3f58d4ed564535982359f2ef159cb8dfc6`.
It contains raw toy-workload logs and must not be described as the public
evidence package.

## What can be checked here

Run:

```bash
python verify_published_evidence.py
```

The verifier checks that:

- three DETAIL records and three automatic Flight Recorder records exist;
- each arm is internally stable;
- operation and input-shape multisets agree across the two evidence sources;
- the Flight Recorder primary belongs to the expected stage-0 DP group
  `[0, 2]`, not merely to an arbitrary two-rank group;
- DETAIL's combined `Float Float` representation and Flight Recorder's split
  `Float` input representation resolve to the same uniform dtype family;
- the automatic records contain completed collectives before the divergence;
  and
- every retained lifecycle record reports no tracked orphan process.

This is a post-run sensitivity check. It was not committed before the GPU run
and must not be presented as a pre-registered gate.

## Retention limitations

The retained archive has lifecycle JSON for DETAIL trials 2 and 3 and all
three automatic-hang trials. DETAIL trial 1 has no retained lifecycle JSON, so
the public package cannot independently prove its cleanup result. The
cross-version opt-in control is retained only as an aggregate three-trial
summary; it cannot be re-derived without the private audit archive.

The experiment directory was untracked at execution time. Protocol revision
`2026-09-12.6` records the author's declared pre-run rules, but Git history
cannot independently establish when those rules were frozen. Future campaigns
must record a committed protocol SHA in every result summary before execution.
