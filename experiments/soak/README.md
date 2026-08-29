# Soak test

## 目标

验证服务在持续混合负载下是否出现内存增长、延迟漂移、吞吐衰减、句柄泄漏或周期性停顿。

## 建议协议

1. 固定模型、服务参数、负载 seed 和请求集合。
2. 预热到 CUDA Graph、allocator 和缓存状态稳定。
3. 运行 2 小时作为冒烟；正式长稳建议 6 小时以上。
4. 每秒采集 runtime DFX timeline，每分钟聚合 latency/throughput。
5. 结束后保留 `malloc_trim` 前后（若适用）、GPU memory 和进程树。

## 判读

- RSS 上升不自动等于泄漏，需结合 PSS、Private Dirty、allocator trim 和存活对象。
- GPU reserved 稳定但 used 波动，通常不能直接判为显存泄漏。
- 只看首尾两点会漏掉周期性锯齿，至少保留完整时间序列。
- 趋势拟合必须剔除 warmup，并同时报告斜率、峰值和最终平台值。

## 2026-08-27 实机结果

10 分钟短稳共完成 1,200 个请求且无失败；预热后 RSS 拟合斜率为 0.96 MiB/hour，GPU memory 采样值保持 20,445 MiB，未观察到明显延迟漂移或抢占。这是 smoke soak，不替代计划中的 2～6 小时长稳。详见 [`../../results/VALIDATION_SUMMARY_2026-08-27.md`](../../results/VALIDATION_SUMMARY_2026-08-27.md)。
