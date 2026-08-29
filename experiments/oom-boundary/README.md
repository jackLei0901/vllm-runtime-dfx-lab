# OOM boundary map

## 目标

把“显存不够”拆成可区分的区域：启动失败、CUDA Graph capture OOM、运行期 KV 压力、抢占退化和真正的运行期 OOM。

## 变量

- `max_model_len`
- `max_num_seqs`
- `max_num_batched_tokens`
- `gpu_memory_utilization`
- 输入/输出长度
- 并发与 arrival rate

先固定模型、dtype、quantization、attention backend 和 CUDA Graph 模式，再一次只改变一组变量。建议使用二分法寻找边界，而不是盲目做完整笛卡尔积。

## 每个点至少记录

- 启动是否成功、启动峰值显存。
- 预热是否成功。
- 请求成功率和错误类型。
- KV cache usage、preemption counter。
- TTFT/TPOT/E2E、吞吐。
- 故障发生阶段和最后一份 incident snapshot。

## 分区

| 区域 | 判据 |
| --- | --- |
| safe | 无错误、无持续抢占，延迟稳定 |
| degraded | 请求完成，但 TTFT/TPOT 或排队明显恶化 |
| preemption | 抢占持续增加，存在可量化重算成本 |
| OOM | 启动、capture 或运行期明确 OOM |

不同阶段的 OOM 不应合并成一个结果。

## 2026-08-27 实机结果

单张 RTX 4090、Qwen3-4B BF16、eager、8K 上下文下，`gpu_memory_utilization=0.40` 因 KV 容量不足被启动检查拒绝，0.41 能正常启动并完成 7,000 + 512 token 请求。该结果定位的是启动容量边界，不是运行期 CUDA OOM。详见 [`../../results/VALIDATION_SUMMARY_2026-08-27.md`](../../results/VALIDATION_SUMMARY_2026-08-27.md)。
