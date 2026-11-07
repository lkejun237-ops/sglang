# DiT 与 SGLang 实例卡顿回归：serving-stall-v1

## 测试目标与边界

固定单活跃会话、真实输入和实际部署配置，回答两件事：输入已经准备好时，DiT
是否出现异常长调用或无故不启动；完整实例是否持续交付结果，或在首请求、长会话、
条件更新及取消恢复时出现长尾。平均 FPS 不能代替逐次停顿检查。

游戏、浏览器、RTC 不进入这两层测试。源输入采用冻结 RGB/conditioning，按帧号关联
输入与输出。线上游戏暂停或断供另做产品链路测试，不从 simulationTick 是否变化
推断服务计算是否卡顿。此规范中的阈值是回归策略，不是已测容量或现网 SLA。

## 固定用例矩阵

所有用例都串行执行，在 **832×480、1280×736** 两档分别出结果；736 必须有单独
冻结并核验的输入，不能沿用 480 档的输入身份或以 720 替代 736。
复用原始录像时，允许按目标尺寸独立解码／缩放，但须保存原视频 SHA、转换命令、
派生 case ID 和 RGB SHA，并明确这是派生输入，不能声称原录像原生为 736。
优先复用 product-case-20261029 的真实请求，保留完整 prompt、参考图顺序及更新边界。

必须分别记录输入 RGB、Source VAE 实际编码、DiT target 与最终输出的尺寸。现有生产
`ZING_SOURCE_RESIZE` 将实时 source 编码限制为短边 480、面积 832×480；因此本轮
736 档实际为输入 RGB 1280×736 → source 编码 832×480 → DiT/输出 1280×736，
capture 中 source latent 为 30×52、target latent 为 46×80。这覆盖完整 736 输出
路径，不是 Source VAE 全尺寸 736 性能证据；不能只用入口尺寸标注所有组件。

| ID | 驱动与持续时间 | 要发现的问题 | 必需证据 |
| --- | --- | --- | --- |
| D1 · 完整 DiT 上下文 | 生产 sampler、完整层数、实际 SP 布局；真实 conditioning 提前准备；预热后 3 轮完整长 case，媒体时间轴至少 10 分钟 | 单次计算长尾、通信等待、CPU 发射间隙 | 完整块调用 Host／异步 CUDA event、逐 rank、shape/step/history、冻结输入 ready barrier／event enqueue／enter／return、历史块淘汰；实际计算墙钟单列，Graph 内部独立 phase 标未测 |
| D1-soak · 常驻 DiT 耐久 | 同一进程／模型连续执行完整 case，正常 EOS／释放后立即开始下一会话；3 轮，每轮所有 rank 的有效会话工作累计至少 600 秒 | 反复建会话后的 GC、缓存增长、资源残留和长尾 | 每次完整会话及释放后的显存、逐块 Host/CUDA、整个常驻轮 GC；会话间回执写盘和协调不计入 600 秒 |
| S1 · 实例供满 | 绕过游戏与 RTC，真实 RGB 尽快送入，严格遵守 credit；每个固定 case 预热 1 次、正式 3 次 | 有积压时调度停顿、Encoder/DiT/Decoder 任一阶段拖慢 | source send/accepted/consumed 水位、逐批 RGB 到达、三阶段逐 rank 计时 |
| S2 · 实例实时输入 | 同一输入按 24 FPS 发送；独立记录计划可用时间与实际发送时间；长会话至少 10 分钟 | 实时播放断续、积压增长、内存/缓存周期性抖动 | 独立输入节奏记录、逐帧输出依赖、虚拟播放缓冲、credit、显存时序 |
| S3 · 冷首请求 | 隔离测试实例受控重启 3 次；每次 ready 后立即发真实请求，再发 3 个相同暖请求 | ready 过早、JIT/懒加载/缓存首次执行长尾 | 重启/ready/首请求时刻、TTFF、prefill、首 3 个输出 block 的阶段记录 |
| S4 · 条件切换与恢复 | 按固定边界执行长 prompt、换图；另在独立请求中测试取消后下一会话 | 更新引起的停顿、旧会话资源残留 | update receipt、首个受影响输出、取消回执、资源释放、下一请求首帧 |

D1 必须保持真实 causal/KV 演进、source/denoise/commit 和生产调度语义，不能以缩层、
随机 tensor、同一个静态 forward 重复跑代替。旧 bench_dit_phases 的 SP1 缩层结果
只能验证工具与组件计时，不能直接作为 D1 的生产布局验收。

媒体时间轴长度与实际计算墙钟分别报告。15921 帧按测试的 24 FPS 计为
663.375 秒内容，一次供满 replay 可能更快完成，不能据此声称单会话持续计算
十分钟。现有 case 通过 D1 验证完整自然上下文，通过 D1-soak 验证常驻进程的
十分钟工作；后者是多个正常结束的完整会话，不把录像循环拼成更长的单一上下文。
需要验证单会话更长的自然上下文时，另冻结更长真实素材，当前范围明确列为未测。

`benchmark_dit_replay.py --minimum-round-seconds 600` 启用 D1-soak。每次完整
会话都重建合法 KV/RNG 状态，沿用同一份冻结输入和生产 schedule；不缩层、不改
步数、不加 sleep、不改为 paced。只有所有 rank 的有效工作时长都达到门槛才停止
追加完整会话，不能按本 rank 的时间独自退出。保存每个已完成会话的进度，失败
直接中止而不重试；回执写盘及轮次协调的时间另列，不能帮助满足工作时长门槛。
整个常驻轮的 GC 观察包括这些间隔，另标注是否与实际会话工作重叠，避免将回执
序列化的 GC 算成模型调用停顿。单次完整回放与常驻耐久的数据分别统计。

长素材不足时，该项标记未执行；不得循环一个短视频并继续沿用原 case 身份。允许
在会话边界重复完整冻结请求做额外 soak，但结果明确标为“多会话串行 soak”，不能
作为“单会话 10 分钟”通过。取消/恢复只操作本轮隔离测试实例，不打断他人会话。

SDK capture hold 的 20 秒暂停/恢复属于独立的产品链路回归，不包含在本套固定输入
测试中。合法 hold 不进入 S1/S2 的连续供帧统计；若测试游戏链路，必须记录
hold/resume 的确认回执，不能事后按 tick 相同剔除样本。

EOS 尾批必须纳入 S1/S2：保留最后两批的到达时刻、帧范围与尾批 underflow，
使用累计播放缓冲计算，不能只用批间隔减上一批时长。默认隔离启动向现有
`--disagg-encoder-ports` / `--disagg-decoder-ports` 各传入唯一自有端口，复用连接，
避免每个会话同步关闭一次性连接后才返回最后 RGB。仍只允许一个活跃会话；
取消、断连、EOS 后必须检查下一会话可用性。不能复用产品服务端口或其他人的 worker。
旧 scalar 端口兼容路径保持原行为，对照时按冻结版本记录，不在运行中切换配置。

终态关闭还须覆盖“最后 RGB／profile ACK 已到达，客户端仍继续发送心跳”的情况。
先取消并 join 会话 listener；收集 profile 时临时消费终态输入，发送 profile 前停止
该 reader，让 ACK 独占接收。最终 CLOSE 握手期间通过原始 ASGI `receive()` 消费
迟到消息，避免单条应用队列阻塞控制帧；关闭完成、异常或取消后必须 join reader。
不得在活跃 listener 存在时启动第二个接收者，也不能把正常 CLOSE 当成请求失败。

连接清理耗时是嵌套的整体边界，不能与子项相加。短请求对照不代替长流、条件切换、
冷容器与生命周期验收；各冻结版本的结果及原始数据回查
[PR #580](https://github.com/seedleap/sglang/pull/580)。

S3 先于 GPU capture 执行。capture 即使使用独立进程，也可能提前填充同一容器中的
磁盘编译缓存。首个容器内推理和已运行过同档输入后的进程重启分别标注；只保留后者
会漏掉首次调用的长等待。未经核验，不声称容器外或镜像内所有缓存均为空。

## 卡顿定义与归因

### 实例交付

使用客户端同一个 monotonic 时钟记录每批完整 RGB 的到达时间与帧范围。模拟以
24 FPS 播放、首包后固定预缓冲 250 ms 的消费者：

```text
首批：play_end = arrival + 250ms + batch_frames / fps
后续：underflow = max(0, arrival - play_end)
      play_end = max(play_end, arrival) + batch_frames / fps
```

记录所有 underflow；单次大于 250 ms 标记交付卡顿，大于 1 秒列为严重事件。
分别报告次数/分钟、累计值、最大值、逐事件位置，并报告 ICL 的 P50/P95/P99/max。
批次间隔超过 1 秒只作诊断线索：首批已经提供足够视频时，长批间隔不等于卡顿。
这是虚拟播放缓冲推演，不是实测浏览器显示延迟。预缓冲配置必须随结果保存，不能
增加缓冲来掩盖回归。原始到达记录和冷 TTFF 始终保留。

交付检测结果 schema v2 中，`underflow_count` 记录所有缺帧事件，`stall_count`
记录严格超过配置门槛（默认 250 ms）的次数，`severe_underflow_count` 只记录严格
超过 1 秒的次数。旧 schema v1 的 `severe_underflow_count` 实际使用配置门槛，不能
将其解释为超过 1 秒；历史数据保留原样，用原始 events.jsonl 重算 v2 后分别报告。

S1 不以播放 underflow 作为唯一门槛；主要检查有 accepted backlog 时的工作进度。
S2 才使用播放门槛，且要求设备已经有同配置 24 FPS 能力证据；算力不足应报告
持续欠速与积压，不能把“稳定低吞吐”描述成偶发抖动。

### DiT 归因

按冷/暖、初始/稳态 block、3/2 phase、step、合法 KV/history 长度、shape 与 rank
分别比较，不能把初始四步和后续两步混合统计。逐次记录 Host/CUDA 耗时及相邻
ready→enter 间隔，保留最大值附近的前后至少 5 秒事件。

相同组内，超过 baseline 中位数 3 倍且额外增加至少 100 ms，作为待归因异常候选；
这个规则不能单独证明计算卡顿，也不能代替匹配条件的 baseline。冷请求单列，
不得因标记“预热”而删除 S3 结果。

当前 D1 的实际边界为整次 `append+drain_ready`，包括生产 Graph；一个 Graph 中
融合的内部 phase／step 没有独立 Host 调用边界，不得用整块 event elapsed 假装
逐步耗时。保留实际采样策略、shape/history、初始四步／稳态两步身份，逐块记录
Host 进入／返回及 event 提交时刻。所有冻结输入提前传到 GPU，run 前同步完成
时刻是 ready 的上界，不是线上 Encoder 的 enqueue 时间；同机时钟只在同一 rank
的记录中解释。历史块前后身份及淘汰差值已记录，allocator reset／compaction 内部
事件没有单独计时。前后集合差记录的是被移除的历史块，可能包括窗口淘汰、prompt
reset 与 EOS dispose，不直接当作窗口 eviction 次数。需要继续区分异常块内部 phase 时，另采 nsys/CUDA Graph 节点
诊断并关联固定 shape/phase；这项内部归因在取得 trace 前保持未测，不改变正式
Graph 路径或用 forward hook 切成 eager 后冒充同条件计时。

| 观测 | 可支持的归因 |
| --- | --- |
| 输入已 ready、排队明显，DiT 没进入 | 调度/发射路径待查 |
| CUDA event elapsed 同时升高 | GPU/通信路径待查；event elapsed 不等于 kernel busy |
| Host 升高、CUDA elapsed 稳定 | CPU/同步/搬运路径待查 |
| DiT 稳定，Encoder/Decoder 增长 | 对应阶段待查 |
| 服务端完成稳定，客户端到达间隙增长 | 发送、网络、客户端解包/写盘待查 |

需要定性 GPU idle/通信/实际 kernel 工作时另采 nsys 诊断轮。正式性能轮不逐调用
同步、不加重 profiler；诊断轮与正式轮的开销分别记录。不同机器 monotonic
时间不能直接相减，跨节点依赖用 request/dispatch/frame ID 关联；单节点内部才算
queue gap，时钟域不明的结果标记无法归因。

DiT replay 默认保留正常自动 GC，并记录 chunk 循环与 finish 内的 `gc_events`，使用
同一进程的 `perf_counter_ns` 与 block Host 边界关联。它只观察，不关闭、延后或强制
收集；state 构造和 dispose 不在该观察区间。CUDA event elapsed 会包含等待 CPU
发射的时间，不能把与 gen2 GC 重合的长块写成“GPU kernel 卡顿”。如做静态对象
冻结对照，必须独立标注 GC 策略、冻结边界、实际 SP、显存与完整生命周期结果，
不能以对照策略替换默认实例的结果或删掉默认策略下的停顿。

静态堆候选修正仅在 Zing 模型加载完成、尚未发布 ready 或开始 replay 时执行，先
collect 再 freeze，记录冻结数量、自动 GC 状态与准备耗时。新请求在此后创建，仍
接受自动 GC；不在活跃会话中再次冻结。worker 结束后解冻，再沿用原关闭与收集流程。
该策略随对应源码版本报告；`SGLANG_ZING_FREEZE_STATIC_GC=0` 是未冻结基线，
不能将两种策略混作同条件。DiT replay 的 `STATIC_GC.json` 保存实际策略，GC 观察器
本身仍只观察。通用非 Zing pipeline 没有该启动 hook，保持原策略。

独立 Encoder 在 load/warmup 后、socket 和 ready 前执行同一准备步骤，ready JSON
保存实际回执；退出或清理失败时关闭句柄。Encoder-only 对照只改变该子进程的
开关，DiT 保持原策略。被动 gen2 诊断按独立开关启用，不改变 GC 阈值或冻结策略；
正式轮和阶段诊断分开。按明确 worker 角色及 session 区分事件，禁止从控制组名称
里的 `encoder` 字样推断实际角色，也不能将日志缺失当作零暂停。

### CUDA Graph 捕获期间的 finalizer 安全性

默认隔离套件在 S3 后执行 `check_cuda_graph_gc.py`，独立 GPU 子进程创建并丢弃
编译内核，在真实 Zing `CudaGraphBackend` 捕获时以低阈值触发自动 GC，核对输出、
调用者 stream 及退出后的 GC 策略。临时编译目录不与模型缓存共享；此项不是 D1
或实例性能结果，也不能代替完整模型回归。`--unprotected` 仅供诊断原生 Torch 捕获，
预期失败也必须保留驱动日志，不作为默认通过路径。

2026-11-06，同镜像 Torch 2.13.0+cu130 单卡诊断中，未保护组两轮均发生
`cuModuleUnload` 错误 900，随后 allocator 检查和 capture_end 报错；延后 GC 两轮
均通过。它复现 [PyTorch #199117](https://github.com/pytorch/pytorch/issues/199117)
的风险，但不能仅凭相同最终异常认定完整实例 736 换图失败由它触发。该实例没有
记录首个 CUDA API 错误，根因仍需带驱动日志复验。

Zing Graph 在 eager warmup 后、捕获前暂时关闭自动 cyclic GC，先结束捕获并恢复
调用者 stream，再恢复原先的 GC 状态；已禁用 GC 的调用者保持禁用。它独立于
启动静态堆冻结开关，不禁止显式 collect 或引用计数析构，不在每个 replay 上切换
策略。新请求环仍可在捕获外收集；异常路径及嵌套状态恢复纳入 CPU 回归。
真实 Zing backend 的独立 GPU 对照已完成：旧版本两轮均先报内核卸载错误 900，
`9bbeb3df` 两轮均成功；刻意 invalidated capture 后 stream/GC 恢复也通过。原始
证据在本轮私有产物的 `capture-gc-backend-r2-9bbeb3df/`，仅验证该故障机制；
480/736 完整模型和实例的新修正验收仍在执行。

### 捕获 stream 与跨线程同步

`9bbeb3df` 已开启上述 GC 保护，旧 scalar 连接的完整矩阵仍在 736 换图请求
第 17 帧后失败。驱动日志记录捕获失效及另一线程的 `cuStreamSynchronize` 错误
900；尚不能仅凭记录顺序指定首因。此前 56 个请求成功，失败请求及剩余未执行项
单列保留，不能将 GC 复现通过等同于所有捕获问题已解决。

[PyTorch 2.13 的 stream pool 契约](https://github.com/pytorch/pytorch/blob/v2.13.0/c10/cuda/CUDAStream.h)
说明：同优先级 stream 按 32 个底层对象循环发放，池由所有线程共享；持有 Python
`Stream` 对象不代表预留了底层 stream。DiT 的长期 capture stream 因而可能与后续
RPC 的短期 stream 重合，`thread_local` capture mode 不会让同一条底层 stream
变成线程私有。此机制与该实例失败的关联仍需实际 stream 身份和调用栈证明。

2026-11-06，同镜像空闲单卡、真实 `9bbeb3df` 受保护 backend 的独立机制对照中，
256 次申请得到 32 条底层 stream，8 个 wrapper 与 capture stream 重合；后台同步
该 stream 报错并使捕获失效。改用 CUDA driver 非池化 stream 后没有重合，同样的
后台同步、捕获及逐位输出检查成功。两策略各一个独立进程，不是完整模型验收，
也不能代替原失败请求的 stream 身份；原始记录在 `stream-alias-r23-r2-9bbeb3df/`。

候选改用 CUDA driver 创建的非池化 capture stream，每个进程／设备仅保留一个，
供串行 Zing 捕获使用。其生命周期覆盖全部 Graph 与库 workspace，由进程退出
回收，不在 Python/CUDA 关闭次序不确定时析构，不随会话数增长。其他 RPC 仍使用
原有 stream。当前候选完整 GPU 验收待执行，不将此代码变化冒称已解决实例失败。

独立Decoder worker也保留自己的非池化stream，按设备复用；它与capture stream使用
不同缓存键，不共享底层handle。现有跨lane step gate继续串行包围decode、publish及
rank agreement，每会话Lynn state、线程、输入event等待和record_stream保持原样；
dispose不销毁进程级stream，旧线程未退出时仍按原quarantine门禁处理。此候选用于
验证多stream allocator缓存累积，不能从reserved增长直接推断泄漏或停顿因果，
也不保证消除cuDNN等库的线程缓存；需要同时检查输出、stream ID、allocator、
RSS、取消／故障恢复和完整端到端指标。当前同条件GPU前后验证未完成。

默认套件另执行 `check_cuda_graph_stream.py`：创建 256 个普通 stream，分别在
真实 denoise／prefix backend 捕获中让另一线程同步普通 stream，核对底层身份
隔离、完整输出、caller stream 恢复和线程退出。它与 GC 用例均在独立 GPU 进程
运行，不进入正式性能计时；同样不能代替两档完整模型、条件切换及生命周期验收。

## 每轮验收与保存

- 修改 Decoder stream 或会话资源回收时，默认追加同一物理 GPU 的跨会话资源对照。
  复用完整短／长 capture，模型常驻、正常 EOS 后开下一会话，记录 allocator、stream
  身份和输出等价性。执行方法见 [Decoder 跨会话资源归因](../../../../../benchmark/zing_h3_causal/STALL_REGRESSION.md#decoder-跨会话资源归因)。
  此项隔离 DiT／网络，仅用于归因，不能替代 S1/S2 或以 reserved 增长直接判定泄漏。
- 每个相关性能 PR 执行匹配的 baseline/candidate，480/736 分开；常驻、冷启动、
  条件更新、取消恢复分别报告。首次接入先冻结硬件与 case 的 SLO。
- 默认提议：S2 暖态不得出现单次 >250 ms 的播放 underflow，>1 秒为严重失败；
  冷 TTFF/prefill/tail drain 单列，使用冻结的硬件级绝对上限与 baseline 比较。
  尚未冻结上限时标记未验收，不能以“平均 FPS 正常”通过。
- DiT 必须取得完整逐 rank、对应 shape/phase 的证据才判定；缺阶段、缺 rank、
  掉采样或只有客户端 JSON 时，DiT 状态为未测/证据不足。
- 记录源码/镜像/checkpoint/config SHA、实际 GPU/拓扑、精度、schedule、并行布局、
  Graph/compile/offload、输入与参考图 SHA、时间范围、请求 ID、warm/cold 身份、
  失败/取消/未执行数。基线与候选不一致则禁止算收益。
- 报告 TTFF、E2E、全程及首包后 FPS、TPOC、ICL、source→output、tail drain、
  credit wait，另给三部分分项耗时与显存。长会话监测 OOM、错误、显存增长、
  输出帧范围连续性和会话结束后下一请求可用性。
- 测试结果至少分为通过、失败、未执行/证据不足；失败请求也保留原始事件。

## 当前工具落地范围

`bench_streaming_source.py` 默认对每请求 events.jsonl 生成 `stall_check`，写入
requests.jsonl/results.json；单独重算用 `analyze_stream_stalls.py`。检测器检查
连续帧范围与完成回执，输出卡顿事件、TTFF、credit 与输入 schedule lag。
输入 lag 可能来自上一轮 credit 背压，不能据此删掉卡顿事件或认定游戏 CPU 故障。

检测器只返回交付层 `no_stall_observed` / `stall_observed` /
`insufficient_evidence`，显式保留 `dit_verdict=not_evaluated`。它不自动给整个
serving-stall-v1 判通过，S1 供满模式也不能套用实时播放门槛。D1 独立 runner、
独立 paced producer、冷重启编排、长输入/736 冻结及完整门禁仍需对应 GPU 验证。

当前入口已支持 product-case-20261029。`run_stall_regression.py` 默认串行运行两档完整 S1 与长会话 S2，预解码 RGB 在计时外按哈希核验。部署生命周期、D1 以及最终归因独立验收，runner 完成不能代替整个规范通过。

隔离测试的模型／任务启动入口使用 Linux subreaper 监督进程，从启动前建立子树
归属。即使实际 launcher 提前退出，独立 session 的孤儿子进程也由该监督进程收养；
正常退出或信号退出都必须在有界时间内终止、等待并 reap 全部自有后代，直到内核
确认没有剩余子进程。入口失败不得启动实际任务，清理失败保持非零退出；实际任务
原有非零状态不能被清理结果覆盖。此机制只用于 benchmark，不改生产 launcher。

外层仅接受原监督进程私有控制管道返回的完整清理回执；该描述符不会传给实际模型
进程。监督进程已退出仍须检查回执，异常死亡、缺失或未完成回执均不得标为清理成功。
不重新解析已 reap 的旧 PID／PGID 来猜测或终止残留进程，避免误伤复用该编号的任务。
单次监督清理预算为 30 秒，短于外层终止升级的 45 秒；只有确认子树清空后才允许
后续会话／测试继续。监督入口的启动开销属于工具版本和冷启动条件，不能直接沿用
旧工具的冷启动数字。

默认隔离布局使用 DiT SP 卡数加 Encoder、Decoder 各一张专卡；SP2 共四张，SP4 共六张。
调度分配与实际使用分别记录：本轮 L20D API 仅允许 0/1/2/4/8 卡分配，SP4 配方
实际申请八卡、使用六卡，另两卡闲置；不能按六卡报告资源成本。使用前核验全部
获配设备空闲，保留角色到 GPU UUID 的映射，不为凑卡修改节点拓扑或停止他人作业。
显式选择共享卡 fallback 时，工具在会话结束和 Qwen
编码结束后释放该进程不用的 CUDA allocator 缓存，并记录释放前后显存；这是共享
测试的内存策略，计入该配置的耗时，不能与专卡结果混称同条件。活跃 tensor、KV
和 CUDA Graph 私有池不会被释放。2026-11-05 的未修正共享基线在 480 长会话发生
OOM 后导致对端 collective 超时；原始失败样本必须保留，不得只重跑成功样本。

独立归档校验必须绑定实际对象路径、分块顺序、字节数和 SHA256。相同内容位于
不同 OSS 路径时，不能仅凭整件 SHA 相同复用另一个路径的回读结果；只有完全相同
且已校验的不可变对象组合才可复用证据。运行中快照须列出尚在写入或未校验的文件，
全部写入结束后再做最终核对；缺少失败请求的客户端 SHA 时明确注明证据边界。
