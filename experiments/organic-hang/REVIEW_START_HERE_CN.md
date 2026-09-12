# 四卡真实 Hang 案例：本地审核入口

## 本阶段要回答的问题

前面的九次双卡实验是在我们自己设计的故障上验证方法。下一阶段要检查：面对一个并非为本项目设计、且已有独立答案的真实上游问题，这套证据链能否从默认 hang 现场恢复出相同的 collective mismatch。

选择的案例是 PyTorch FSDP2 条件参数问题：不同 rank 使用不同参数，反向阶段形成不同的梯度集合，最终让 reduce-scatter 的数量或形状不一致。开启 PyTorch `DebugLevel.DETAIL` 时会直接报告 collective mismatch；关闭后可能表现为 hang。

## 为什么不是先找 vLLM Hang

默认 vLLM 的稳态 TP collective 大量经过 custom all-reduce 或 pynccl，而不是 ProcessGroupNCCL。PyTorch Flight Recorder 因此可能只看到初始化阶段，无法提供完整的逻辑序号。直接拿这类问题验证，会把“证据源看不到”与“诊断方法无效”混在一起。

FSDP2 的 collective 经过 c10d，适合先验证 Flight Recorder 与外部栈的组合是否真的能处理一个自然产生的问题。vLLM TP=2 仍保留为后续集成与边界实验。

## 已经冻结的内容

1. 原始 issue、转帖、复现 Gist 和修复 commit 已逐项核对。
2. 原始源码 SHA-256 和改写后源码 SHA-256 均已固定。
3. 改写器只增加 timeout、端口、DETAIL 开关、统一关闭 anomaly detection、observer 授权和 rank PID 输出；没有改变模型、随机分支、seed、PP/DP 拓扑或梯度逻辑。统一关闭 anomaly detection 是为了保证 A/B 只相差 DETAIL，而不是同时改变反向同步行为。
4. 四个实验臂、成功条件、失败条件和数据保留边界已经预注册。
5. 结果 Schema 已要求 oracle 与 Flight Recorder 同时提供结构化 `primary_divergence`；verifier 比较 operation、子组规模以及无序的 input shape/dtype 多重集合。DETAIL 的局部 rank/sequence 与 Flight Recorder 的全局 group/sequence 分别保留供审计，但不会被错误地当成同一身份空间。
6. 正式 A、B、D 各三次；Arm C 是独立的三次 manual-dump 试验，不能复用 timeout 后已退出的 Arm B 进程，也不能补救 Gate B。
7. 四卡 preflight 会检查 GPU 数量与型号、精确 PyTorch 版本、NCCL、驱动、py-spy 和 prepared-source hash，并嵌入每份结果摘要。

## 建议审核顺序

1. [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md)：编号、源码、修复和版本边界是否可靠。
2. [`fetch_and_prepare_reproducer.py`](fetch_and_prepare_reproducer.py)：实际 diff 是否只包含声明的观察性改动。
3. [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md)：oracle、三次重复和 GO/NO-GO 是否在看结果前足够明确。
4. [`organic-hang-result-v1.schema.json`](organic-hang-result-v1.schema.json)：结果是否有封闭、可验证的数据边界。
5. [`parse_detail_oracle.py`](parse_detail_oracle.py)：是否只保留 mismatch 所需字段，不保存原始日志。
6. [`verify_organic_results.py`](verify_organic_results.py)：是否可能在缺少四份 FR、稳定 signature 或 fixed control 时误判通过。
7. [`preflight.py`](preflight.py)：是否能够阻止错误版本、错误 GPU 拓扑或被修改的 target 进入正式实验。
8. [`LOCAL_GATE_RESULT_2026-09-10.md`](LOCAL_GATE_RESULT_2026-09-10.md)：本地门禁、实际通过项与首次上机 smoke 边界。

## 当前尚未完成

本地门禁已经完成三项：真实 CPU/Gloo DETAIL fixtures、四 rank
PID/start-time 生命周期保护，以及受影响/修复版 CUDA 13.0 wheel 的可安装性核对。
因此本地阶段已经允许进入租卡。以下两项在租卡后、读取正式结果前闭环：

- Flight Recorder 的四 rank、多 process-group、shape 和 thread identity
  已有合成 fixture 覆盖；仍需用 PyTorch 2.11 产生的真实 pickle 验证 decoder 输入契约。
- Arm C 的 `thread_id`、py-spy `os_thread_id` 与 `/proc/.../comm` join
  已有单元测试；真实 autograd issuing thread 仍需在四卡 Arm C 中确认。
- 在 Linux 上对真实四 rank 子进程树执行一次非正式 cleanup smoke，确认实现与单元测试一致；
- 用 PyTorch 2.11 生成一份真实 Flight Recorder pickle，确认 decoder
  看到的 process-group、shape 和 issuing-thread 字段符合冻结契约。

已冻结的解析边界不再依赖作者预先知道的函数名：组 key 使用全局成员、p2p 标记和逻辑 sequence；rank 内只按 `record_id` 排序；项目 frame 必须来自 prepared target 的精确完整路径。

## 当前本地结论

review 揭示的核心 P0 已修正；协议 `2026-09-10.4` 进一步把 60 秒 timeout 应用到 DeviceMesh 创建的所有子进程组，并依据真实 pickle 明确区分 DETAIL 的局部身份与 Flight Recorder 的全局身份。只有在上述 CPU gate 全部通过后才进入四卡租用；GPU 只回答真实拓扑复现、自动 dump 与 cleanup 行为。
