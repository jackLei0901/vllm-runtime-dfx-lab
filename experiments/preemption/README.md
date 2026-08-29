# Preemption cost

## 目标

回答抢占不仅“发生了多少次”，还付出了多少重计算、排队和吞吐代价。

## 最小对照

- 低压力负载：理论上不触发抢占。
- 边界负载：偶发抢占。
- 高压力负载：稳定触发抢占。
- candidate：只改变被验证的调度或 KV 行为。

## 指标

- `num_preemptions_total` 增量。
- running/waiting 请求时间线。
- TTFT、TPOT、E2E P50/P95/P99。
- output throughput 和成功率。
- 如果代码路径可提供，记录 recomputed tokens；不能提供时不得用抢占次数替代重算量。

## 判读

修复抢占风暴不等于所有 latency 都会改善。保守的 stop-and-wait 方案可能减少无效状态迁移，但增加单个请求的等待时间；必须分别报告机制指标与用户指标。

## 2026-08-27 实机结果

在相同的 8 × (7,000 input + 512 output) 负载下，将 KV cache 从 11.59 GiB 降到 6 GiB 后观察到一次抢占：output throughput 下降 37.72%，P99 TTFT 变为 3.88 倍，而 mean TPOT 基本不变。每个条件目前只有一次运行，结论用于确认机制和采集链路，不作为稳定的性能均值。详见 [`../../results/VALIDATION_SUMMARY_2026-08-27.md`](../../results/VALIDATION_SUMMARY_2026-08-27.md)。
