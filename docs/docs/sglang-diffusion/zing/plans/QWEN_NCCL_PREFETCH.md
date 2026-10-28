# Qwen 提前编码与 NCCL 结果接收、流水线 CLI

最初基于 main `7add23fd798f0eaf4a99a8ac452074a63a9db0e4` 独立适配；后续合并 main `5e66d5e2c3`（已含 #417 与 #445），不额外修改 #417 的调度设计。
目标是在旧 prompt 的 DiT 仍运行时，对已收到的新 prompt/reference 提前执行 Qwen，并通过 NCCL 接收 GPU 结果。
此前 main 在 NCCL 模式跳过预取器，直到正式更新边界才编码；本次接通已有预取器和设备结果接口。

## 使用

同机独立 Qwen 服务与 DiT 的部署入口现在默认使用 NCCL，无须在两条启动命令或两份 JSON 中重复开启。保留原有 token、native、limits 和身份 profile。显式 CLI > JSON `result_transport` > 默认 `nccl`；旧 JSON 明确写了 `"result_transport": "http"` 时仍使用 HTTP，删除该字段才会采用新默认。需要 HTTP 时，两端显式选择 `--conditioning-result-transport http`。
在既有 remote 启动命令上增加 `--zing-conditioning-prefetch`，保持单活跃会话：

```bash
sglang serve --model-path MODEL_PATH \
  --zing-conditioning-mode remote --zing-conditioning-prefetch \
  --conditioning-service-url http://127.0.0.1:8093 \
  --conditioning-config CLIENT_CONFIG \
  --conditioning-expected-profile PROFILE \
  --realtime-max-sessions 1 --realtime-max-sessions-per-worker 1
```

此片段需与实际 GPU 布局及模型参数合并。Qwen 服务仍监听 loopback，使用相同 profile；升级部署按原 build-profile 流程校验/重建代码身份。
默认 NCCL 要求 NVIDIA CUDA、同机 IPv4 loopback 和单接收 worker；不满足时显式使用 HTTP，不静默回退。首次真实 DiT 请求预热 NCCL 建组。HTTP 仍传输入和控制信息，结果 context/row_role 走 NCCL。
不开预取保持同步 NCCL；HTTP 预取保持 CPU 结果。提前编码仍默认关闭；仅部署入口的传输默认值由 HTTP 改为 NCCL，不增加预取生命周期总开关。底层 `ConditioningClient` / `ConditioningService` 的 Python 构造器保留 HTTP 兼容默认，部署入口始终显式传入已解析的传输方式。

## 独立开关统一使用命令行（本轮扩展）

每项能力保留独立开关，不增加总开关、preset 或新的配置文件。复用 `ServerArgs` / `StoreBoolean`；布尔参数接受裸参数（true）或显式 `true` / `false`。新参数默认 `None`，未传时沿用旧 profile、环境变量或 JSON；显式 CLI 覆盖旧配置，`sglang serve --config` 沿用现有 CLI > 配置文件优先级。模型权重、鉴权、limits 和代码身份 profile 仍是已有配置，不改为命令行秘密参数。

| 独立能力 | DiT 命令行 | 兼容输入 / 未配置时行为 |
| --- | --- | --- |
| Qwen 提前编码 | `--zing-conditioning-prefetch` | 原 CLI，默认 false |
| Qwen 结果传输 | `--conditioning-result-transport nccl` | client JSON `result_transport`，未配置时默认 nccl；Qwen `serve` 同名参数覆盖 service JSON |
| Source Encoder 并行于 DiT | `--zing-source-encoder-mode overlap` | profile `source_encoder.remote.mode`；需已有远程 Encoder |
| Source 跨 dispatch 预取 | `--zing-source-encoder-prefetch` | profile `source_encoder.remote.cross_dispatch_prefetch`；需 overlap |
| Worker 异步输出 | `--zing-source-async-output` | profile `async_source_output`，默认 false |
| Decoder 与下一块 DiT overlap | `--zing-source-decoder-overlap` | 原 CLI，默认 false；需独立 Decoder、quantum 1、单会话 |
| 每次 dispatch 推进步数上限 | `--zing-source-work-quantum 1` | `SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM`，默认 0 |
| 基础联合 forward | `--zing-source-joint-forward cross_chunk` | profile `joint_forward`，未配置时 cross_chunk；也可 off / source_first |
| 实验性延迟 final / prompt joint | `--zing-source-defer-final` / `--zing-source-prompt-joint` | 对应 `SGLANG_ZING_SOURCE_DEFER_FINAL` / `SGLANG_ZING_SOURCE_PROMPT_JOINT`，默认 false；按需独立启用 |
| 分卡角色 | `--encoder-gpus` / `--denoiser-gpus` / `--decoder-gpus` | 已有 CLI；不新增 Qwen 启动器 |

以下是 Linux / NVIDIA CUDA 同机四卡布局的流水线参数示例：DiT 使用可见卡 0、1，Source Encoder 使用卡 2，Decoder 使用卡 3；Qwen 独立进程在服务 JSON 的 `native.gpu_id` 选择卡 2（两进程均启动于同一原始四卡可见列表）。其余模型、算子、显存及采样配置沿用已验证部署，不能直接根据此片段认定任意硬件容量足够。Qwen 六项算子优化与 FA4 / Graph 不属于本次入口统一范围。

```bash
# 先启动 Qwen，等待 ready；SERVICE_CONFIG 含 token、native、limits。
python -m sglang.multimodal_gen.runtime.conditioning.cli serve \
  --config SERVICE_CONFIG --profile PROFILE

# 在另一终端启动 DiT 和已有 Encoder / Decoder role launcher。
sglang serve --model-path MODEL_PATH \
  --num-gpus 2 --ulysses-degree 2 \
  --encoder-gpus 2 --denoiser-gpus 0,1 --decoder-gpus 3 \
  --zing-conditioning-mode remote \
  --conditioning-service-url http://127.0.0.1:8093 \
  --conditioning-config CLIENT_CONFIG --conditioning-expected-profile PROFILE \
  --zing-conditioning-prefetch \
  --zing-source-encoder-mode overlap --zing-source-encoder-prefetch \
  --zing-source-async-output --zing-source-decoder-overlap \
  --zing-source-work-quantum 1 \
  --realtime-max-sessions 1 --realtime-max-sessions-per-worker 1
```

需要同时打开续推三分支和 prompt 切换六分支时，再加：

```bash
--zing-source-joint-forward cross_chunk \
--zing-source-work-quantum 1 \
--zing-source-defer-final --zing-source-prompt-joint
```

`defer-final` 对应上一块 final + 下一块 source + 下一块 target 首步；要求原有五 latent history、video-only、无 CFG、after_step0 和 q1，不满足继续按原准入报错。`prompt-joint` 对应 conditions + history 1/2/2 + source + target 首步；是否命中要看 `forward_counts.prompt_joint` / fallback 指标，不把开关值当作性能证据。

`async-output` 把输出物化放到 worker 后台；`decoder-overlap` 允许 API 在上一块 RGB 尚未收回时推进下一块，二者仍是独立策略。`encoder-prefetch` 让已接收的 source 在 dispatch 间进入远程 Encoder；`encoder-mode overlap` 控制远程 Encoder 的执行模式。关闭某一项可在同一命令中设 `--zing-source-encoder-prefetch false` 等，恢复 HTTP 则两端都设 `--conditioning-result-transport http`。

API 先应用现有角色端口，再只覆盖上述部署策略；不修改磁盘 profile 的 hash、solver、去噪步数或容量。API 把已解析 quantum / 实验策略写入内部 worker command，避免 API 与 worker 环境不一致。worker 不从用户输入采纳这些策略。Qwen transport 在两端各自入口解析，已有握手检查仍拒绝不一致。

验证计划：真实参数解析到 ServerArgs、profile 不变且显式 false 生效、旧环境回退、内部 command 序列化到 worker 后策略一致、Qwen 两端 CLI 覆盖 JSON、非法组合和握手错误。本轮两轮完整 CPU 测试已通过，结果见末节；480/736 GPU 验证缺口仍不变。只扩展配置入口，不合入 #417 的调度行为，也不扩展 FA4 / Graph 等算子开关。

## 调用链和所有权

1. API 的既有 publisher 将 prompt/reference 候选发往本会话 IPC 通道，保留一个正在处理的任务和最新待处理消息。
2. `ZingSourceStreamState._start_conditioning_prefetch` 仅在 SP leader 主线程取得显式 CUDA device，创建预取器。
3. `ConditioningPrefetch` 共用单锁和单候选；后台与边界 fallback 都按结果协议调用 HTTP `encode` 或 NCCL `encode_device`。
4. `NcclReceiver` 复用独立通信组和私有 CUDA stream，完成 GPU 接收、stream 同步及 HTTP 完成确认之后才返回；预取器此时才能发布候选。
5. 正式边界由 `RemoteConditioner.__call__` 重新校验实际输入 digest，各 SP rank 协调后复用候选；未完成则等待剩余时间，未命中则在同一锁下执行 fallback。
6. 主线程 `_allocate` 为接收张量登记消费 stream，随后沿既有 SP 广播，成功后替换 active condition。后台不调用 SP collective、不修改 active condition/KV。

不另建 GPU 队列、线程池或跨 rank 后台广播。已有接收器完成同步足以形成发布屏障，无须再加一套 event 协议。
`record_stream` 防止 GPU 候选被替换、Python 引用释放后，allocator 在消费 stream 尚未完成时复用存储；它不替代接收完成同步。
正在使用的 active condition、单候选和接收中的结果各有原所有者，故“单候选”不等于全进程仅一份结果显存。

## 超时、取消与失败

边界的锁等待、输入序列化和 HTTP 控制共用请求 deadline，不在设备接口重新获得完整预算。CPU 参考图准备发生在编码调用前，不承诺被此 deadline 抢占。
首次握手增加可选 `connect_timeout_s`，从客户端剩余预算取得；服务端排队和 TCPStore 连接使用该连接预算。原 `timeout_s` 保留持久 NCCL 组的操作超时，连接完成后恢复 Store 的正常操作超时，避免一次慢准备把后续所有请求的通信超时永久缩短。新服务兼容省略连接预算的旧握手；新客户端发送新增字段，需要两端一起升级。
这不是整个调用严格按 deadline 返回的保证：已进入的 NCCL 操作、CUDA stream 同步及安全排空可能超过请求 deadline，仍须持有 live buffer，失败后通道不可复用。不得用后台遗弃收发或提前释放存储来兑现硬超时。
输入准备或序列化已经耗尽预算时，在接收器入口拒绝，不污染健康通道；开始通信后的不确定失败沿用 #415 的失效通道策略，拒绝复用，需重启 worker/Qwen 服务。
取消时先停止接纳，正在进行的接收排空后丢弃结果，不发布迟到候选；未排空时保留 live owner 和异常，禁止新会话抢用。
接收失败的 GPU buffer 仍由失效通道持有。HTTP 输入校验失败允许边界重新校验；不把 NCCL 失败重试为第二轮不确定收发。

## Review 修正：消费同步与连接预算

消费候选后只同步当前消费 stream，以保留分配/H2D 完成和错误协调，再执行原 SP 广播。NCCL 候选已在接收 stream 完成同步后发布，无须在消费方再次同步整个 device；否则候选 A 消费时可能等待后台候选 B 的接收。`record_stream` 仍保护消费期存储。
选择现有 stream 的同步而不新增 event/完成队列，既保留 HTTP 和非 leader 分配路径的就绪保证，也不会主动等待无依赖的预取 stream。
验证：实际 worker 入口在 CUDA 替身下消费候选时检查只同步消费 stream、随后广播，禁止全卡屏障；HTTP/NCCL 均覆盖。首次连接检查客户端/服务端剩余预算、服务端排队过期、旧握手兼容和下一请求正常操作超时恢复。定向42项通过；完整默认 CPU 套件两轮各1888项通过、59子测试通过、0失败/跳过（589.70s / 386.44s）；Linux工具238项通过。原始命令/JUnit/日志为 `D:/qwen-nccl-prefetch-20261028/review-fix-full-*`，源码绑定回执为 `review-fix-validation-receipt.json`。这些是控制流和设备替身证据，GPU 数值、实际并发和性能仍需单独实测。

## 验证计划与证据

| 不变量 | 检查方式与通过条件 | 本轮状态 |
| --- | --- | --- |
| 真正提前启动，边界不重复编码 | 真实 IPC、受控设备接口，阻塞后台接收并验证前台等待/复用 | CPU 回归通过 |
| 输入变化不能误用旧结果 | prompt/reference digest、latest-only、延迟消息、关闭/失败路径，HTTP/NCCL 参数化 | CPU 回归通过 |
| leader 接收，主线程 SP 消费 | session 启动与 RemoteConditioner 入口回归，记录线程和 device | CPU 回归通过 |
| deadline 不重置、失败不误复用 | 序列化/锁等待/建组/接收阶段探针与现有 NCCL 测试 | CPU 回归通过 |
| 跨 stream 存储安全 | 消费 stream 登记测试；真实 CUDA/NCCL 生命周期仍需 GPU 验证 | CPU 登记契约通过，GPU 未测 |

旧本地实验 `e545049fda` 在 B 卡四卡 480 三例实现过组合路径，每例提前编码一次命中、零 fallback；它包含旧 #417 与额外配置，不能作为本次 main 适配的 GPU 验收或性能结论。
当前版本尚未测试 480/736 的 GPU 正确性、性能和故障恢复。本轮先执行本地 conditioning/source 回归；不自动申请云资源。
后续 GPU 验证按 PERFORMANCE_DEFAULTS 和现有 streaming benchmark，固定模型、输入、布局、采样、缓冲及计时，分别测 480/736 的 HTTP 预取、同步 NCCL、NCCL 预取。
报告 TTFF、prompt/reference 发送至首新 RGB、FPS、队列、Qwen/reference VAE/source VAE/DiT/Decoder Host-CUDA P50/P95 与逐卡显存；共享 GPU 的竞争和首次建组成本单独记录，不将异步等同于免费并行。

## 实践依据与 reviewer 导读

复用本仓库 #410 的会话 IPC、digest 和主线程 SP 契约，以及 #415 的 NCCL receiver/超时失效处理。
2026-10-28 核查上游 main（当日解析为 `990ee89877f668304761d7551d803ed6b93d1673`）的 [CUDA IPC transport](https://github.com/sgl-project/sglang/blob/990ee89877f668304761d7551d803ed6b93d1673/python/sglang/srt/multimodal/transport/cuda_ipc.py)：其生产就绪和消费完成分别管理，映射在消费 stream 完成前保持持有。本适配借鉴生命周期边界，继续复用已有 NCCL 接收器，不移植 IPC 内存池。
[PyTorch 2.14 record_stream](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.record_stream.html) 用于防止跨 stream 消费结束前回收 tensor 存储；本实现选择登记消费 stream，避免另增回收队列。
这是本仓库 Zing 扩展，不能声称是上游已经支持的 Qwen 功能。
本仓库 [Diffusion 贡献指南](../../contributing.mdx) 要求精简注释/异常包装、提交前验证并给出性能比较；本轮复用既有预取/通信模块，配置入口复用 ServerArgs、StoreBoolean 和 API 的部署 profile overlay，不引入新的调度层或配置框架。真实 GPU/E2E 尚未验证，仍是社区合入前的证据缺口。
配置部分遵循本分支 `ServerArgs.from_cli_args` 的 CLI > server config 合并和 `StoreBoolean` 的显式 false 语义；部署 profile 保持磁盘不可变，仅在 API 内存中覆盖。配置职责在原入口，不让张量计算层反向解析 CLI。
审查重点：预取与 fallback 是否只提交一次收发；候选是否在接收完成后才可见；SP 广播是否始终由原调用线程执行；deadline 和取消是否保持通道所有权。
阅读顺序：`source_session._start_conditioning_prefetch` → `ConditioningPrefetch._encode/encode/_receive` → `ConditioningClient.encode_device` → `NcclReceiver.encode/_exchange` → `RemoteConditioner.__call__/_allocate` → `test_prefetch.py`、`test_nccl.py`；配置再看 `ServerArgs.add_cli_args` → `ZingH3SourceStreamAdapter.on_init/prepare_next_request` → `ZingSourceStreamState.initialize`，以及 `conditioning.cli.load_configuration` → `RemoteConditioner.from_server_args` 和 `test_pipeline_cli.py`。

## 2026-10-28 本轮验证结果

实现基于上述 main 的独立分支 `codex/qwen-nccl-prefetch`；未修改 #417，也未部署；已发布 Draft [PR #452](https://github.com/seedleap/sglang/pull/452)。
WSL / Python 3.12 / PyTorch 2.14.1+cpu，使用仓库 `benchmark/zing_h3_causal/run_conditioning_cpu_tests.py` 的默认 13 组入口（显式传入对应文件/目录以保存 JUnit）。
完整一轮为 1855 passed、1 failed、0 skipped，另 56 subtests passed（368.50s）。唯一失败是历史源码对照测试的 `git show` 无法解析 Windows worktree 的绝对 Git 路径，未进入功能断言。
仅给该项重跑设置正确的 `GIT_DIR`/`GIT_WORK_TREE` 后，1 passed（11.85s）；未更改该测试、业务代码或跳过规则。合计 1856 个主用例有通过证据，不能表述为同一次全绿运行。
其中 `test_prefetch.py` 31 项、`test_nccl.py` 26 项均在完整轮通过；HTTP/NCCL 路由、真实 IPC 等待/复用/取消、失败通道拒绝复用、剩余 deadline、leader-only 初始化及消费 stream 登记均覆盖。
真实网络测试用 loopback HTTP，张量往返包含 CPU Gloo；CUDA 接收/stream 生命周期测试使用替身，不证明真实 NCCL/GPU 并发。
首轮定向检查曾因三项新测试使用简写 digest 而失败，已修正为有效 digest 后完成上述完整回归；没有放宽业务校验。
isort、Ruff format、Ruff E9/F63/F7/F82/F401/UP037、`git diff --check` 均通过。

本地原始命令、环境、日志及两份 JUnit 保存在 `D:/qwen-nccl-prefetch-20261028/`：
`validation-command.json`、`regression.log`、`regression.xml`、`git-path-recheck.xml`。
作者自审沿 IPC → prefetch → client → receiver → worker/SP 检查方案必要性、状态和存储所有权：保留原单候选即可，未发现需要新增队列或独立调度器的依据；检查范围内未发现阻塞问题。
未进行独立子代理 review，也未执行当前提交的 480/736 GPU/E2E 或故障恢复验收；旧四卡实验不能替代这些缺口。


## 流水线 CLI 扩展验证（2026-10-28）

两轮完整 conditioning 默认套件均为 **1876 passed、0 failed、0 skipped，另 59 subtests passed**；用时363.49s / 413.66s。WSL Linux / Python3.12 / torch2.14.1+cpu，139项依赖版本通过仓库 conditioning_recipe 校验。每轮包括新增的 CLI/config 优先级、显式 false、profile 不可变、joint 三种模式、API command 与 worker 语义、Qwen 两端传输覆盖及既有 IPC/Source/取消故障回归。与受测源码的 SHA256 绑定记录位于 `D:/qwen-nccl-prefetch-20261028/pipeline-validation-receipt.json`，命令、日志及 JUnit 为同目录 `pipeline-full-{1,2}*`。

在此之前定向回归初轮910通过3失败（新测试的 serve config 参数和通用 request factory 夹具缺项），修正夹具后915通过；再增加 joint 三模式用例后执行上述完整双轮。不修改业务校验或跳过测试。新增文件已按双轮证据登记 CI，既有受影响测试证据同步刷新。isort/Ruff/diff 检查通过。

作者自审确认：直接补齐现有入口即可，无须新配置文件、总开关或统一调度器；CLI → ServerArgs → API 内存 profile / 内部 command → worker，以及 service/client transport 两端均已追踪。GPU stream/collective 的实测缺口沿用前述限制；CPU 合成组件通过不证明 480/736 数值、吞吐或真实异步收益。本轮未运行 #417、云任务或部署。

## 默认 NCCL 的后续调整（2026-10-28）

为避免在 Qwen / DiT 重复开启传输，仅修改已有部署入口的缺省值，不引入协议协商或自动回退。CLI 仍为 None，以保留显式 JSON 配置；HTTP 模式的外部 TLS 测试明确选择 HTTP。测试实际经过 Qwen `cli.main` 的配置解析和 service 构造参数、DiT `RemoteConditioner.from_server_args` 到客户端的解析，覆盖缺省 NCCL、JSON HTTP/NCCL、CLI HTTP 以及 CLI 覆盖 JSON，磁盘配置保持不变。

两轮七模块定向 CPU 回归各 173 passed、27 subtests passed，0 failed / 0 skipped（47.26s / 43.39s）。命令、日志、JUnit 和源码摘要：`D:/qwen-nccl-prefetch-20261028/nccl-default-*`。本轮不是完整套件；GPU/480/736 性能未测，前述完整回归属于此前提交。


## 2026-10-28 四卡 W6 实测与 main 冲突整合

GPU 受测提交为 `6421f08610`：上海 L20Z/SM90 四卡、W6、832×480、360帧一次冷流程完成；提前编码命中1次、失败/回退0次，实际通过NCCL接收CUDA结果并由两个DiT rank消费。Qwen Host调用482.37ms，提前编码全调用543.47ms，相对消费方提前150.88ms，仍等待392.69ms；发送开始到接收返回1.55ms（含Host/控制开销）。六分支joint forward执行1次，首次warmup未Graph回放。TTFF53.718s，首批后交付17.00FPS，不作为预热稳态或A/B收益。配置、计时边界、显存、限制及原始证据见[完整实验记录](../../../../../benchmark/zing_h3_causal/QWEN_NCCL_PREFETCH_RESULTS.md)。

本轮合并 main `5e66d5e2c3` 的唯一文本冲突位于 Source 初始化策略：保留 main 的 `_source_key_policy` 共用函数，并让正常会话与 Block0 prefill 都优先读取内部 command 的 `work_quantum/defer_final/prompt_joint`，缺省才读取旧环境变量。缓存键包含实际解析策略，避免预填充与正式会话配置不同导致错误复用或无法命中。新增回归从两个真实入口追至缓存 lookup，验证相同命令产生相同key、显式false覆盖环境变量、策略变化产生不同key。GPU结果属于合并前提交，合并后只执行针对受影响路径的CPU验证，不外推GPU结论。


合并验证：受影响的8个测试模块共 **1006 passed、3 subtests passed、0 failed、0 skipped**（pytest 60.36s），覆盖CLI解析、NCCL/预取、worker消费、Source入口/会话、preparation cache与Decoder overlap；新增两组缓存键一致性回归通过。isort、Ruff检查/格式与git diff检查通过。原始命令与JUnit：`D:/qwen-nccl-prefetch-20261028/merge-main-command.json`、`merge-main-targeted.xml`。首次尝试把依赖模型测试环境的`test_zing_prompt_switch_cache.py`加入conditioning专用runner时，因runner的registry替身缺少`get_model_info`而在收集阶段失败；该模块未计入上述通过范围，也没有将它从仓库CI移除。模型级Block0缓存回归未在此runner执行，未重跑GPU。


### 合并后的 benchmark CI 修复

`f805abc03a` 的 benchmark-contracts 两轮均在 `test_source_work_quantum.py` 的两组续推用例失败：AST测试夹具只提取真实方法、跳过构造器，漏填新增的 `source_defer_final/source_prompt_joint` 属性。正式初始化已正确设置这些字段。修复仅为测试夹具补齐两个false默认值，不改变产品代码或降低原断言。WSL CPU定向复测该文件 **7 passed**（1.27s），isort/Ruff/格式及diff检查通过；原CI日志留在 `D:/qwen-nccl-prefetch-20261028/ci-f805-benchmark/`，复测JUnit为 `ci-fix-quantum.xml`。
