# Issue #249：持续 source 流水线与 GPU 验收

本文为 Issue #249 冻结版本的历史实验记录，保留当时的 W6/S3、
`split_first_latent=true` 配置和原始测量。当前缺省窗口为 `21/2`，请求与
KV pool 的启动 window/sink 必须一致；当前模型与动态 Sink 约束见
[动态 Sink 规格](dynamic-window-sink.zh-CN.md)。下述历史结果不作为当前默认的验收结论。

历史实验分支隔离依赖 PR #176，冻结提交
`8f75d4c49840c5cab4d8ab33f82ec15ddb255e79`；当时 #176 尚未合并，
该 Draft PR 不满足合并准入。下文“本轮”均指该历史实验。

可信 deployment profile 的 `source_encoder.remote.cross_dispatch_prefetch=true`
仅适用于 `mode=overlap`。API 的私有同节点 IPC 将已接受的真实 RGB 按原 packet
顺序交给 Encoder producer；GPU dispatch 继续独立验证原始 source 身份、区间与
ledger credit。不同会话使用随机、私有目录中的独立 IPC 地址。
当前部署仅支持一个活跃会话；API admission 与 Encoder 均限制为单会话，
不提供多会话容量开关。第二会话被拒绝时不修改现有 owner。
producer 不执行 DiT collective、不修改 DiT 进程的后端策略；沿用两个 moments
credit。API ledger、消息上限、输入帧数和字节容量继续限制缓冲；不释放额外 credit。

矩阵输入显式设置 `cross_dispatch_prefetch=true`、`async_source_output=true` 和
`source_chunk_plan={"num_frame_per_block":[3,2],"split_first_latent":true}`。
新增流水线仅在 overlap 臂启用；shared 与 dedicated serial 保留同步基线，三臂分块相同。
`progressive_decode=false` 保留现有 TAEH3 默认解码，避免混用未通过 exact 的实验策略。

`async_source_output=true` 将 TAEH3 分段输出交给现有两槽 RGB materializer，
每结果最多 128 MiB，保留分段顺序、精确 RGB24 转换、CUDA copy event 和 session/generation 身份校验。
慢消费者占满槽时等待或失败，不静默降为额外同步拷贝。
Decoder 与 DiT 保持同一卡组；没有实现新的 Decoder。

本轮业务基线要求 `num_frame_per_block=[3,2]`、`split_first_latent=true`，
实际序列是 `1,2,2,3,2,3,2,…`。不是循环五项数组。
测试必须冻结真实 source+六图、四步、音频关闭、performance/确定性开关及真实路由。
GPU 仅 ap-southeast-7 现有 shengdong profile/quota，单节点最多两卡；
CPU 准备与后处理零 GPU，所有配置串行且有超时；产物保持私有。

## 容量实验与默认窗口失败

冻结提交 `8da62b0f39792a333e54c5f8adcfdb7a6ddefba0` 在真实双卡上完成
480p Encoder 的 102 项 moments/latent 严格比较（atol=0、rtol=0）。
历史 shared 基线当时默认 window=21、sink=3，在 dispatch_index=10（此前已完成索引 0–9）时 OOM；
此前已接受 154 帧、输出 141 帧。KV 的动态存储由约 47 GiB 增至 94 GiB，
下一次几何扩容失败；失败位于 `ZingH3KVCache._reserve_store`。
硬件为 PyTorch 可见总显存 274,113 MiB/卡、CC 10.3。
该失败不能记为完整视频、流水线收益或实时验收通过。

该历史容量实验在可信 `zing_realtime_profile.json` 中显式指定
`source_attention_window={"window_chunks":6,"sink_chunks":3}`，矩阵输入使用
同名字段。该冻结版本未指定时为 21/3；当前缺省为 21/2，请求与 KV pool
的启动 Sink 必须同步配置。客户端 window/sink 覆盖仍被拒绝。
三个臂均使用同一个窗口，正确性、正式测量、profile 和长流的绑定包含该字段，
不能复用不同历史覆盖的结果。6/3 是单独的容量实验，改变 retained latent
和历史覆盖；不与默认窗口或旧分块混算，也不把这部分收益归于预取。

## 当前验证

- 真实本地 IPC 的 CPU 控制回归证明 scheduler accepted watermark 仍为 1 时，
  producer 已处理后续 8 帧；两槽满时不会产生第三份 moments。
- worker 单 owner 的 busy 拒绝、关闭/abort 后下一会话及故障隔离由 CPU 回归覆盖。
- 动态 Encoder scope 只对实际依赖高于 dispatch 水位的 latent 归因；
  未来 source 到达但当前编码不需要它时不会误计为跨 dispatch overlap。
- 这些使用 CPU 故障注入，不证明真实 CUDA overlap、精度、吞吐或显存验收。

GPU 实现源码冻结为 `8346a674aa40ed203948a159ba66703e4bc8a473`。
同一真实 source、六参考图、四步与音频关闭配置下，window=6/sink=3 的
832×480、896×512、1280×736 严格矩阵均完成。每尺寸 Encoder 的
102 项 moments/latent 精确一致；shared、E1 serial、E1 overlap 各输出
完整 720 帧，source/moments/latent/noise/DiT/Decoder/RGB 两组全链路
比较均通过。旧分块的 480 对照也单独完成三臂严格比较。
这些不是历史默认 21/3 或当前默认 21/2 窗口的通过结论。

480 新分块完成三臂、满速与 24FPS paced、各五轮正式测量。
overlap 满速/paced 中位分别 17.57/17.50 FPS，相对 serial 耗时改善
33.70%/31.98%，block、per-frame 与 dependency P95 均改善，完整 RGB
五轮配对一致。满速 serial 的五轮耗时/P95稳定性未通过独立保守
检查（极差/中位数不超过 10%）；该检查不是统计置信区间。
核心中位/P95门槛与稳定性检查分别报告。仍低于持续 24 FPS，
paced 输入有积压，新开关继续默认关闭。

480 的 serial paced、overlap paced、overlap unpaced 120 秒输入均完成
预热与测量各 2880 帧，三组完整 RGB 精确一致，EOS 后输入文件与调度
资源清理成功。源素材为原始 120 帧业务视频循环，不代表自然长视频
接缝无瑕疵；媒体中可见循环边界的融合变化，音频未验收。

独立 R3 profile 保留 CPU/CUDA 事件与关联 ID，仅采样 DiT 的启动、早期、
晚期及 EOS dispatch，Encoder trace 覆盖整个 profile 会话。
按实际 dispatch 水位与下一次提交裁剪，并扣除时钟误差后，480 找到
9 条 Encoder–DiT CUDA 重叠 witness，合计 266.51725 ms。Encoder–Decoder
观测重叠为 0 ms，不能声称其重叠通过。实际 Perfetto 已打开 DiT 原始
trace 与明确标注范围的 Encoder ROI，核对业务 scope、CUDA kernel、
设备、stream 与关联 ID。原始 trace 完整保留；ROI 只供人工查看。
Perfetto 的 complete-event 重叠导入告警影响嵌套布局，查看器说明未丢
事件，仍保留此限制。profile 耗时不作为正式性能。

512 新分块完成三臂、满速/paced、各五轮正式测量及 serial/overlap
生命周期门槛。overlap 满速/paced 中位 15.2910/15.2940 FPS，相对
serial 耗时改善 33.3065%/33.2988%，核心中位与三类 P95 门槛通过，
五轮完整 RGB 配对精确一致；两组极差/中位数≤10%的稳定性检查均
失败。五轮满速配对 bootstrap 的收益 95% 百分位区间约 25.97–33.89%，
P95 差值区间仍包含退化可能；这是顺序执行五轮的探索性诊断，不构成
随机交错实验的因果置信结论。仍未达到持续 24 FPS，paced 有积压。

512 六组独立 profile 完整 RGB exact。完整 Encoder 7,470,551 事件与采样
DiT 的实际水位对齐分析找到 9 条 CUDA overlap witness，合计
165.9685 ms；Encoder–Decoder 观测 0 ms。实际 Perfetto 已核对 DiT
原始 trace 与明确标注的 Encoder ROI，业务 scope 内 kernel 的设备、
stream、correlation 与时间区间可见。DiT/ROI 的 complete-event spill
布局告警分别 1366/557，UI 说明未丢事件，原始 trace 完整保留。

512 overlap paced 120 秒预热与测量各 2880 帧，完整 RGB 一致，EOS
清理成功；测量 block P95 约 57.06 秒，仍有持续输入积压。30/120 秒
媒体原始哈希、完整帧数与全文件解码通过；各 20 个声明边界帧已采样
审查，循环附近有纹理/平台几何变化，不宣称自然长视频无接缝。

736 新分块完成三臂、满速/paced、各五轮正式测量及 serial/overlap
生命周期检查。overlap 满速/paced 中位 7.2686/7.2703 FPS，相对匹配
serial 耗时改善 30.8169%/30.9187%，核心 P95 与独立五轮稳定性检查通过；
仍低于 24 FPS，paced block P95 约 66.78 秒。分配重试告警保留，
完整输出不代替显存余量验收。

736 独立 profile 中 Encoder 13,859,927 个原始事件完整保留；9 条
Encoder–DiT CUDA witness 合计 1008.4485 ms，Encoder–Decoder 为 0 ms。
实际 Perfetto 已核对 DiT、Encoder ROI 和 EOS 的 CPU launch/CUDA
correlation。EOS 时间窗包含先前排队的 DiT kernel，不能将窗口内所有
kernel 耗时归因于 Decoder；关联 launch 的 EOS CUDA union 为 15.601661 ms。
RGB24 materialization/D2H 缺少独立业务 scope，成本未知，不能记为零。

三尺寸 serial paced、overlap paced、overlap unpaced 的九组长流均完成
预热及测量各 2880 帧、完整 RGB 一致、EOS 清理成功。736 测量 paced
block P95：serial 421.70 秒、overlap 256.37 秒；120 秒是输入时长，
不是完成时限或持续实时通过结论。480 长流对照来自不同物理 GPU 对，
不作为同设备配对因果性能比较。

九组长流逐会话/逐 rank 资源审查中，输入 ACK 峰值 68 帧、末尾待 ACK
为零；Encoder pending result 不超过 2。736 的 KV 在前段 source 35 帧
扩容，随后稳定；后段同类快照 allocated 波动不超过约 1.08 MiB，reserved
不增长。EOS 释放会话 KV/codec/bridge/metadata，固定模型权重继续驻留。
5 秒驱动采样并非瞬时峰值；736 overlap unpaced 最低观测空闲 18,430 MiB。
主机 RSS 求和可能重复计算共享页，cgroup memory.current 不可用；
空闲 pinned pool 不代表全部活跃传输内存。退出时 semaphore 告警保留。

真实慢消费者（每输出 batch 延迟 4 秒）在预热阶段收到 39 帧后触发
keepalive timeout；没有完整 720 帧结果，不能记为无丢帧或完整 exact 通过。
写出阻塞约 10.86/12.01 秒。API 输入文件已清理，scheduler release 失败；
客户端失败后 harness 立即关闭服务，因此这是 teardown 条件下的负面清理
证据，不能直接断言正常运行服务存在永久泄漏。

旧分块 480 正式六端点各五轮完成，30轮完整 RGB exact，serial/overlap
共20项生命周期通过。overlap 满速/paced 为13.9077/13.7722 FPS，
相对匹配 serial 耗时改善28.1529%/28.4500%；核心门槛通过，满速
五轮稳定性通过、paced稳定性失败，仍未达到24 FPS。
旧分块六组独立 profile 完整 RGB exact；完整 Encoder 5,998,304 个事件
与九个采样 DiT trace 哈希验证。15 条 Encoder–DiT witness 合计
440.52725 ms，Encoder–Decoder 观测 0 ms。实际 Perfetto 已核对 DiT、
future Encoder 与 EOS 的 CPU/CUDA/correlation；DiT/ROI/EOS 的布局
spill 告警为 1040/561/1109，TraceErrors/DataLosses 为零。

以下为旧双会话实现的历史诊断，不属于当前部署支持范围或验收要求。
三尺寸补充并发隔离诊断均通过：seed42 的17帧 A、seed63 的34帧 B，
并发与各自独立运行完整和逐帧 exact，两个 seed 的输出确有区别；
每尺寸四个独立会话均清理。短输入隔离诊断不替代30秒业务准确性基线。
第三会话 busy-owner 保护沿用独立 CPU 控制回归证据，不能将此双会话
GPU exact 检查表述为额外的第三会话 GPU 测试。

原故障 harness 误杀 launcher 而留下独立进程组中的 rank，该无效记录
完整保留。修正版在实际 SID 的 chunk3 RPC inflight 且已有客户端输出时，
按 PID/create_time/命令核对并终止真实 Encoder rank。客户端收到
“Video generation failed. Please start a new session.”；约120秒RPC失败
等待后，API输入文件与scheduler释放成功，DiT会话状态dispose为零。
身份保护清理确认无剩余自有活跃进程；两个已停止zombie和退出semaphore
告警保留。worker已死后的remote cleanup错误仍有记录；不声称服务自动恢复，
不把RPC inflight等同CUDA kernel已开始，客户端错误码未直接落盘观察。

全部 GPU 阶段与最终零GPU全量分析完成。120轮正式输出、80项生命周期、
九组长流、六份媒体完整RAW逐帧全零扫描及四组Perfetto ROI哈希复核完成；
六份媒体无全零帧。测试执行完成与所有启用门槛通过分别记录，后者为否。
所有实验开关保持默认关闭，PR 保持 Draft。

数值 exact 的比较对象为冻结 SGLang 的 shared/serial/overlap 三臂，
不等同于新增 native Zing 基准或完整感知质量验收。长流源视频由120帧
原素材重复，媒体边界变化与音频关闭分别记录。模型66文件本轮复核
既有全SHA回执及路径/大小/mtime，未重新全内容哈希。

最终八组探索性五索引bootstrap各枚举3125次；512满速三类P95差值
范围跨零，旧480 paced dependency P95范围跨零。顺序执行并非随机交错，
区间不代表因果置信，不能覆盖稳定性、慢消费者或24FPS失败门槛。
最终汇总JSON SHA256为
`8cd724129501e6982623bdcacacca0eae04319edbc8f5e9384d9e552eccc7705`。
GPU实现源码仍为8346a674aa40ed203948a159ba66703e4bc8a473，后续验收
文档提交不代表GPU运行过新的实现源码。完整媒体与证据仅保留私有CPFS/OSS。

最终 paced CUDA 水位核对：新480/512/736与旧480的 Encoder–DiT
保守重叠分别248.037 / 268.603 / 671.38425 / 275.64725 ms，
witness数9 / 9 / 10 / 15。上述人工Perfetto主审查为满速单元，
其Encoder–Decoder均0 ms；736 paced另观测单条0.0015 ms，
相对于84.48 µs时钟对齐误差极小，不据此声称Decoder重叠性能通过。
四组正式与profile使用同一GPU UUID对GPU-29809b0e/GPU-32fd29f2，
与早期480长流不同设备的限制分别保留。
