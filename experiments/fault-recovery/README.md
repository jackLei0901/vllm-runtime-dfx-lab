# Fault recovery

## 已完成案例：#48966 / PR #52178

目标是验证“异常 EngineCore 死亡必须让顶层进程非零退出，而 SIGTERM 正常停服仍退出 0”。每次试验都先等待 `/health=200`，再要求一次 completion 返回 HTTP 200，随后注入信号。

### 单卡配对结果

| 路径 | 注入 | 顶层退出码 |
| --- | --- | ---: |
| baseline | SIGKILL EngineCore | 0 |
| patched | SIGKILL EngineCore | 1 |
| patched | SIGTERM API Server | 0 |

### TP=2 结果

两张 RTX 4090 上完成三种进程级验证：

| 注入 | 健康请求 | 顶层退出码 | 孤儿进程 |
| --- | ---: | ---: | --- |
| SIGKILL `VllmWorker-0` | 1 | 1 | 无 |
| SIGKILL EngineCore | 1 | 1 | 无 |
| SIGTERM API Server | 1 | 0 | 无 |

原始证据位于工作总区 `vllm_validation_20260820/52178/`。DP supervisor 具有独立父进程生命周期，不包含在上述结论中。

## 用统一工具重跑

```bash
vllm-dfx snapshot-env --output results/48966/environment.json
vllm-dfx record --pid "$API_PID" --output results/48966 --stop-on-incident
vllm-dfx inject-signal --pid "$ENGINE_PID" --signal SIGKILL \
  --event-log results/48966/injections.jsonl
```

需要额外由 launcher 脚本记录顶层退出码、完整进程树和重启时间。Incident recorder 的职责是补齐故障前状态，不替代进程级断言。

