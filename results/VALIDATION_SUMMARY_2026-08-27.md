# RTX 4090 validation summary — 2026-08-27

This report records the first GPU validation of the Runtime DFX lab. It separates observed evidence from conclusions that still require longer or repeated runs.

## Environment

- GPU: 1 × NVIDIA GeForce RTX 4090
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- vLLM: `0.26.1rc1.dev1103+g7ca49fbe4`
- vLLM commit: `7ca49fbe4bab019e55d57cdc4b7fd3d55c67c1a6`
- Model: Qwen3-4B-Instruct-2507, BF16
- Common server settings: eager mode, `max_model_len=8192`, `max_num_seqs=64`, `max_num_batched_tokens=8192`

The public environment snapshot hashes the GPU UUID. Benchmark-generated text is replaced with `<redacted>`.

## 1. Startup capacity boundary

Only `gpu_memory_utilization` changed. The model, dtype, backend, execution mode and maximum sequence length remained fixed.

| GPU memory utilization | Available KV cache | Result |
| ---: | ---: | --- |
| 0.35 | 0 GiB | Startup rejected: `No available memory for cache blocks` |
| 0.40 | 1.01 GiB | Startup rejected: 8K context needs about 1.12 GiB; estimated maximum length 7,344 |
| 0.41 | 1.25 GiB / 9,072 tokens | Healthy; one 7,000-input + 512-output request completed |

At 0.41, the smoke request returned successfully in 9.43 s, with TTFT 612.39 ms and TPOT 17.26 ms. In this pinned environment the startup boundary is therefore between 0.40 and 0.41.

This is a **startup capacity validation boundary**, not a reproduced runtime CUDA OOM. vLLM rejected unsafe configurations before serving traffic.

![Startup capacity boundary](vllm-dfx-public-20260827-v1/figures/oom-boundary.png)

## 2. Paired preemption-cost experiment

Both runs used the same eight requests, seed and generation parameters: 7,000 input tokens + 512 output tokens per request, concurrency 8, temperature 0 and `ignore_eos`. Only the KV-cache capacity changed.

| Metric | 11.59 GiB KV / 84,416 tokens | 6.00 GiB KV / 43,680 tokens |
| --- | ---: | ---: |
| Successful / failed requests | 8 / 0 | 8 / 0 |
| Preemption delta | 0 | 1 |
| Duration | 15.11 s | 24.27 s |
| Output throughput | 271.00 tok/s | 168.79 tok/s |
| Mean TTFT | 2.32 s | 5.02 s |
| P99 TTFT | 3.82 s | 14.80 s |
| Mean TPOT | 24.92 ms | 24.21 ms |
| Maximum KV usage | below pressure threshold | 99.60% |

With the smaller KV cache, output throughput fell 37.72%, total duration rose 60.55%, and P99 TTFT became 3.88× the high-KV run. Mean TPOT did not regress. For this workload, the visible cost appeared primarily in scheduling/queueing and time-to-first-token rather than steady decode cost.

The recorder emitted both `kv_pressure` and `preemption_storm` warning incidents. A separate 16-request pressure run reached 99.66% KV usage and also observed one preemption.

This is one paired run per condition. It demonstrates the mechanism and validates the DFX capture path, but it is not a statistically stable performance claim. A publication-quality comparison should repeat each condition 3–5 times.

![Paired preemption cost](vllm-dfx-public-20260827-v1/figures/preemption-cost.png)

## 3. Ten-minute short soak

After warmup, the server processed 1,200 requests at 2 requests/s with concurrency 8, 512 input tokens and 128 output tokens.

| Metric | Result |
| --- | ---: |
| Completed / failed | 1,200 / 0 |
| Duration | 601.86 s |
| Request throughput | 1.994 req/s |
| Output throughput | 255.21 tok/s |
| Mean / P99 TTFT | 63.30 / 102.28 ms |
| Mean / P99 TPOT | 15.46 / 20.94 ms |
| GPU memory min / max | 20,445 / 20,445 MiB |
| KV usage max | 5.76% |
| Preemptions | 0 |
| Server errors / fatal traces | 0 |

The API process RSS reached a plateau after warmup. Fitting only samples after the first 60 seconds gives 0.96 MiB/hour; GPU memory remained exactly 20,445 MiB in the sampled timeline. The first and last 300 requests showed no material latency drift. SIGTERM produced exit code 0, with no remaining vLLM or GPU process.

This result means **no obvious short-term leak or latency drift was observed**. Ten minutes is not sufficient to prove long-term stability; the planned 2–6 hour run and allocator/PSS analysis remain necessary.

![Ten-minute soak](vllm-dfx-public-20260827-v1/figures/soak-curve.png)

## Evidence and integrity

- Raw public evidence: `vllm-dfx-public-20260827-v1/`
- Reproducible plotting source: `../plot_validation.py`
- Archive: `vllm-dfx-validation-20260827.tgz`
- Archive SHA-256: `48C68A93EA2EB34D709B9591D0CC5580041FA4F8509B68E10BB91E550D988ADB`
- Manifest: 50 entries, 0 hash mismatches after local extraction
- Lab unit tests: 9 passed locally and on the GPU instance

## What is and is not established

Established in this environment:

- the startup capacity boundary for the pinned BF16/eager/8K configuration;
- a causal paired case in which smaller KV capacity produced a preemption warning and large TTFT/throughput degradation;
- correct incident capture for KV pressure and preemption;
- stable behavior in a ten-minute, 1,200-request smoke soak.

Not yet established:

- a true runtime CUDA OOM after the service becomes ready;
- long-term memory stability over 2–6 hours;
- variance across repeated trials, models, GPUs or graph mode;
- TP/DP behavior for these three experiments;
- NCCL stalls, supervisor restart and end-to-end MTTR.
