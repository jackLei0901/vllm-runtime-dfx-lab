# Four-GPU execution runbook

This runbook is the rental-time boundary for the PyTorch #158719 / TorchTitan
#2747 known-answer reconstruction. Do not start a formal arm until the preflight
for that arm prints `"status": "PASS"`.

## Frozen environment pair

Both Linux CPython 3.12 CUDA 13.0 wheels were confirmed present in the official
PyTorch `cu130` index on 2026-09-10:

| Arm | Exact package | Purpose |
| --- | --- | --- |
| A/B/C | `torch==2.11.0+cu130` | affected build, before fix `775500a5` |
| D | `torch==2.13.0+cu130` | cross-version opt-in control containing `775500a5` |

Use four identical visible GPUs. The model is small; capacity is not the
selection constraint. Topology and reliable NCCL behavior matter more than GPU
class.

```bash
python3.12 -m venv /root/organic-211
/root/organic-211/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130'
/root/organic-211/bin/pip install \
  'numpy==2.2.6' 'py-spy==0.4.2' 'jsonschema==4.26.0'

python3.12 -m venv /root/organic-213
/root/organic-213/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0+cu130'
/root/organic-213/bin/pip install \
  'numpy==2.2.6' 'py-spy==0.4.2' 'jsonschema==4.26.0'
```

Do not share one environment and replace Torch between arms. Separate venvs
preserve the affected/fixed provenance and make reruns auditable.

## Source preparation

```bash
/root/organic-211/bin/python \
  experiments/organic-hang/fetch_and_prepare_reproducer.py \
  --output /root/organic-work/pp_fsdp_graph_test.prepared.py

/root/organic-213/bin/python \
  experiments/organic-hang/fetch_and_prepare_reproducer.py \
  --enable-upstream-fix \
  --output /root/organic-work/pp_fsdp_graph_test.fixed.py
```

The command must print prepared SHA-256
`47fce815563b5ce8e94f93e18f1184b73fa0382ccd3934b7df02a90590c0a5a2`.
The fixed-control command must print prepared SHA-256
`430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1`.

## Preflight

Run once per venv after setting `CUDA_VISIBLE_DEVICES=0,1,2,3`. Record the exact
GPU name and driver from the first run, then pass those values unchanged to all
subsequent formal trials.

```bash
PATH=/root/organic-211/bin:$PATH CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /root/organic-211/bin/python \
  experiments/organic-hang/preflight.py \
  --prepared-target /root/organic-work/pp_fsdp_graph_test.prepared.py \
  --expected-torch-version '2.11.0+cu130' \
  --expected-cuda-version '13.0'

PATH=/root/organic-213/bin:$PATH CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /root/organic-213/bin/python \
  experiments/organic-hang/preflight.py \
  --prepared-target /root/organic-work/pp_fsdp_graph_test.fixed.py \
  --expected-prepared-sha256 \
  '430509263ad2c66786d63d0fa6936c0919bb45e8bf9984ffcb69fde5865c38d1' \
  --expected-torch-version '2.13.0+cu130' \
  --expected-cuda-version '13.0'
```

The second preflight must report the same GPU model and driver as the first.
Changing either after a formal result requires a new environment record.

## Fixed execution order

1. Arm A, DETAIL oracle, affected venv: three trials.
2. Arm B, automatic timeout Flight Recorder, affected venv: three trials.
3. Evaluate Gates A and B. Do not use Arm C to rescue a failed Gate B.
4. Arm C, separate manual-dump and native-stack runs: either absent or three
   trials.
5. Arm D, cross-version opt-in venv with DETAIL disabled: three trials.
6. Run `verify_organic_results.py` only after all mandatory summaries exist.

Every trial uses a fresh state directory and rendezvous port. The process-group
leader is started in a new POSIX session. Rank PID files are bound to Linux
`/proc/<pid>/stat` start times before any signal is sent. A trial has a hard
wall-clock deadline; cleanup sends TERM and then KILL only while the captured
PID/start-time identity still matches. A reused PID is never signalled.

## Retention boundary

Raw stderr, Flight Recorder pickles and py-spy JSON remain in a private audit
directory. Publish only schema-valid summaries, preflight output and aggregate
verification. Record the private archive hash before generating a separate
derived-only public package. Do not claim raw deletion unless the private audit
copy was actually destroyed.
