# vLLM Runtime DFX Lab

这个实验室把线上推理服务的模糊故障转化为可配置负载、可重复故障、统一证据和明确结论。它不是 vLLM 的替代监控系统，而是一套用于复现、PR 验证和技术复盘的轻量工具。

配套文章：[《从能运行到可诊断：搭一套 vLLM Runtime DFX 实验室》](article/vLLM-Runtime-DFX-从故障信号到诊断证据.md)。架构图的 Excalidraw 源文件、实测曲线和绘图代码均在本仓库中。

当前版本实现了第一阶段闭环：环境快照、健康和 Prometheus 指标采集、进程/GPU 状态采集、有界历史、故障触发快照、信号注入、隐私脱敏和 Markdown 摘要。只使用 Python 标准库，远程实例无需额外安装依赖。

## DFX 闭环

```text
Detect 发现异常
  → Capture 保留故障前的有限历史
  → Diagnose 用对照组和时间线定位
  → Recover 验证退出与拉起语义
  → Verify 用回归实验固定结论
```

## 四类实验

1. `oom-boundary`：区分安全区、延迟退化区、抢占区和 OOM 区。
2. `soak`：观察 RSS/PSS、GPU memory、FD、延迟和吞吐随时间的变化。
3. `preemption`：量化抢占次数、重计算 token、P99 和吞吐之间的关系。
4. `fault-recovery`：验证 API Server、EngineCore、worker 的 signal、exit code、孤儿进程和恢复时间。

各目录提供实验定义和判读边界。2026-08-27 已在单张 RTX 4090 上完成启动容量边界、成对抢占对照和 10 分钟短稳验证；`fault-recovery` 另有单卡与 TP=2 证据。完整结果、原始数据边界与复现实验说明见 [`results/VALIDATION_SUMMARY_2026-08-27.md`](results/VALIDATION_SUMMARY_2026-08-27.md)。

## 安装与自检

```bash
cd AI-Infra-Transition/03-stability-lab
python -m pip install -e .
python -m unittest discover -s tests -v
vllm-dfx --help
```

也可以不安装：

```bash
PYTHONPATH=src python -m dfxlab --help
```

## 最小使用流程

先保存环境：

```bash
vllm-dfx snapshot-env --output results/run-001/environment.json
```

服务启动并通过 `/health` 后，记录最近 300 个采样点：

```bash
vllm-dfx record \
  --base-url http://127.0.0.1:8000 \
  --pid "$API_SERVER_PID" \
  --interval 1 \
  --history 300 \
  --output results/run-001
```

在另一个终端注入故障。PID 必须由实验者显式提供，工具不会模糊匹配或自动选择进程：

```bash
vllm-dfx inject-signal \
  --pid "$ENGINE_CORE_PID" \
  --signal SIGKILL \
  --event-log results/run-001/injections.jsonl
```

将 incident JSON 转成可阅读摘要：

```bash
vllm-dfx summarize \
  --input results/run-001/incident-2026-01-01T00-00-00+00-00.json \
  --output results/run-001/incident-summary.md
```

实机验证曲线由 `plot_validation.py` 从原始 JSON/JSONL 重新计算。核心采集工具只依赖标准库；绘图脚本另需 Matplotlib：

```bash
python plot_validation.py \
  --results results/vllm-dfx-public-20260827-v1 \
  --output results/vllm-dfx-public-20260827-v1/figures
```

第一次操作真实进程时，先使用 `inject-signal --dry-run` 核对目标 PID 和事件记录。

## 输出结构

```text
results/run-001/
├── environment.json       # 软件、硬件和 commit
├── timeline.jsonl         # 每次采样的结构化时间线
├── injections.jsonl       # 故障注入事件
├── incident-*.json        # 触发时保存的有界历史
├── incident-summary.md    # 自动摘要
└── run-summary.json       # 本次采集的结束状态
```

采样器默认只保留一组低基数指标，不采集 prompt、token ID 或 structured-output schema。`schema.redact()` 会再次处理常见敏感字段，但它不是通用 DLP 系统；加入新字段时仍需进行隐私审查。

## 统一实验流程

```text
environment snapshot
→ launch server
→ health/readiness
→ warmup
→ workload
→ optional fault injection
→ collect metrics and process state
→ stop/cleanup
→ classify result
→ save raw data and summary
```

## 最小数据要求

- vLLM commit、Python、Torch、CUDA、driver、GPU 和模型。
- 完整 server/workload 命令与 random seed。
- 请求数、并发、输入/输出长度、arrival rate。
- TTFT、TPOT、E2E、wall、吞吐和错误率。
- API/EngineCore/worker PID、exit code 和 signal。
- RSS/PSS、GPU memory、KV usage 和 preemption counter。
- 故障注入和服务退出的统一时间线。
- baseline、candidate、正向控制和负向控制。

## 判读原则

- 观察到相关性不等于确认根因。
- 性能对比必须保证模型、参数、负载字节、并发和版本开关一致。
- `health=200` 只能证明探针响应，不能单独证明引擎仍能完成推理。
- 单卡通过不能外推 DP/TP 的所有故障域。
- 自动摘要只描述采样到的状态，最终结论必须结合日志、配置和对照组。

## 当前证据

- #43444：包含 100 documents/request 的负载与 latency 原始结果。
- #43639：包含 MessageQueue、RSS/PSS/Private Dirty 和 online traffic probes。
- #48966 / PR #52178：包含单卡与 TP=2 的进程级故障注入、退出码和孤儿进程检查。
- RTX 4090 Runtime DFX：包含 0.40～0.41 启动容量边界、相同负载下的 KV 抢占对照，以及 1,200 请求的 10 分钟短稳曲线。
- 统一 CLI 与 schema：本目录实现。

RTX 4090 的公开环境、原始时间线、server log、benchmark JSON、图片与 SHA-256 清单位于 [`results/vllm-dfx-public-20260827-v1`](results/vllm-dfx-public-20260827-v1)，结论边界见 [`VALIDATION_SUMMARY_2026-08-27.md`](results/VALIDATION_SUMMARY_2026-08-27.md)。

## 已知边界

- 当前工具从 vLLM 外部采集，采样周期无法还原每一次 scheduler iteration。
- Windows 只能判断指定 PID 是否存在；Linux `/proc` 额外提供 RSS、VMS 和线程数。
- `nvidia-smi` 不存在或超时时，GPU 列表为空，但其他采集继续执行。
- DP supervisor、NCCL stall 和自动重启恢复时间还没有形成真实 GPU 证据。
- 当前 OOM 结果是启动阶段的主动容量校验，不是服务 ready 后的 CUDA OOM。
- 抢占性能对比每个条件只运行了一次；需要 3～5 次重复实验才能形成稳健性能结论。
- 10 分钟短稳只能排除明显的短期漂移，不能替代 2～6 小时长稳。
