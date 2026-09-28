# Issue #195：performance auto 真机结果

## 2026-09-28 补充实验（采集与技术验收完成）

数值门槛仍失败，默认保持关闭。用户授权继续收集 SP1/SP2 正确性、
正式 A/B B/A 计时、逐卡 trace、服务和 **原始 5 秒** 画质对照；
这不改变 `max_abs <= 0.25` 且 `relative_l2 <= 0.005` 的诊断标准。
`--report-numerical-only` 仅记录 finite、同格式的数值超限并继续独立实验，
跨卡逐字节一致性、未来帧不变性、生命周期和异常恢复仍是硬门槛。

- GPU 作业：`dlc1jyipsa915yrf`，单节点两卡；源码
  `707653b15e8acc062611770858d691e5d0025273`，生产实现未改变。
- 后续同步 main 的冲突修复不改写本轮冻结源码。本文 GPU 数字只对应上述
  `707653b1` 与 padding/batching 关闭的配置，不能当作合并后 main 新默认策略的
  整体性能或组合画质验收；合并后的实现另跑 CPU 回归。
- 环境版本和 GPU 规格同下方首轮记录；本轮 UUID 固定为
  `b84cf848-9fc1-6bf1-bbf0-3a81e5c2bc43`、`127be9f7-32db-9261-0879-5877b0ffc413`。
- 本轮资产清单 SHA256：
  `4d8d1b2d7f8b6ffc3aa79e3ccb569ab76ea093bcf758604a69843aa31681170c`；
  69 个模型文件与首轮逐文件哈希一致。旧 CPFS 实验目录已被清理，
  从保留的源码、私有 OSS 输入和持久权重重建到独立目录；旧失败证据不改写。
- 新增验收器测试在零 GPU 作业 `dlc19yrpqi77cmgr` 中通过：13 tests。
  对应源码的 `taeh3-contracts` CI 通过。
- SP1/SP2 完整 ABBA、80 个服务请求、56 份正式/profile/画质输出均完成技术验收。
  数值门槛仍失败，人工画质仍待用户判断；技术验收完成不代表默认采用。

### 同步 main 后的 CPU 回归

同步 main `04369a33f9` 时保留 performance 生命周期与 bounded tile batching
两组分布式测试，统一使用干净的 forkserver worker，并覆盖 batching 在
`deterministic` / `performance` 两模式下的分工、异形 tile 回退和 exact 结果。
自动合并生成的重复 `local_tile_shapes` 回执字段已去重。
零 GPU 作业 `dlc1msv5e225iy3n` 完成 **1066 passed**（78.44 s），包括完整 tile
矩阵、progressive kernel、spatial padding、media contract、streaming codecs 及
performance 验收器。ruff、isort 和差异空白检查通过。
首轮测试启动器使用 stdin，34 个 forkserver 用例因无法恢复 `<stdin>` 主程序失败；
改用带入口保护的独立文件后全量通过，未为此更改生产实现。
这些 CPU 检查不替代同步 main 后组合默认策略的 GPU 性能或画质验收。

### 已完成的正确性检查

每个配置、每个 rank 均完成 140 个 EOS/分包案例。SP1/SP2 deterministic
均逐字节一致；performance 的实际 gathered moments 和 normalized posterior
在 SP2 两 rank 之间全部逐字节一致。所有配置的未来帧不变性、生命周期和
异常恢复通过；SP2 覆盖 kernel、state、ended、dispose、dtype 五种故障注入。

| 对比（140 案例的最大值） | moments max abs | moments relative L2 | posterior max abs | posterior relative L2 |
| --- | ---: | ---: | ---: | ---: |
| SP1 performance 对同模式 replicated | 0 | 0 | 0 | 0 |
| SP1 performance 对 deterministic | 0.833984375 | 0.0026628071 | 0.5322265625 | 0.0048799273 |
| SP2 performance 对 SP1 performance | 0.48828125 | 0.0023131408 | 0.53515625 | 0.0039971176 |
| SP2 performance 对 deterministic | 0.833984375 | 0.0025767297 | 0.5298843384 | 0.0047013792 |

SP2 performance 对各 rank 的同模式 replicated 也存在数值超限：moments
最大绝对误差分别为 0.486328125 / 0.9411621094；posterior 分别为
0.5336914063 / 0.4091796875。跨卡输出一致不能替代与基准数值接近。
另外核对 reference hash：performance 的每卡 replicated 参考 moments 和
posterior 在 140/140 例中均跨卡不逐字节一致，而实际 gathered 输出始终一致。
deterministic 的参考和实际输出均跨卡逐字节一致。这解释了不同 rank 的参考
并非同一组字节，不改变原数值门槛失败判定。

### 已完成的 SP1/SP2 encoder 正式计时

同卡、独立进程 A/B B/A，每模式每长度 6 个无 profiler 样本，表中为 median。
冷启动和预热不计入，240/1440 帧为原始 120 帧的 2/12 次重放，
用于性能测试，不代表自然长视频质量。
这里比较相同 SP 的 deterministic auto 与 performance auto。SP1 是后端策略
基准；实验开关实际改变的是 performance 下 SP2/4 的 auto 分片。
这些数字不是在相同 performance 模式下单独切换布尔开关的隔离收益，
也没有把 SP1→SP2 增卡收益混入其中。

| 配置与输入长度 | deterministic | performance | 耗时减少 | 速度比 |
| --- | ---: | ---: | ---: | ---: |
| SP1 · 10 s / 240 帧 | 9.413965 s | 8.639861 s | 8.22% | 1.0896× |
| SP1 · 60 s / 1440 帧 | 54.938119 s | 50.437693 s | 8.19% | 1.0892× |
| SP2 · 10 s / 240 帧 | 4.924168 s | 4.538837 s | 7.83% | 1.0849× |
| SP2 · 60 s / 1440 帧 | 28.702236 s | 26.465725 s | 7.79% | 1.0845× |

SP1 首个 latent 分别从 94.22/94.55 ms 降至 83.65/83.60 ms；
输出间隔 p95 分别从 276.44/276.89 ms 降至 255.37/255.62 ms。
SP2 首个 latent 中位数的 performance/deterministic 比值分别为 0.9214/0.7668，
每次输出间隔 p95 的中位数比值为 0.9266/0.9270，通过预先采用的中位数门槛。
SP2 首 latent 存在较大波动：10 秒配置 deterministic 范围 50.32–57.74 ms、
performance 45.03–102.69 ms；60 秒配置分别为 50.61–109.00 ms、
44.95–86.23 ms。6 个样本的 p95 等于最大值，10 秒配置的尾部未改善，
不能把中位数通过写成所有延迟分位均无退化。
两模式的峰值 allocated 显存相同：SP1 16.543 GiB，SP2 每卡 13.390 GiB，
10/60 秒保持相同，所有正式流 dispose 后 retained bytes 为 0。
这些是 encoder 范围计时与显存，不能当作完整服务结果。

### SP1/SP2 完整服务计时与原始 5 秒输出

八个独立服务进程按每个 SP 的 A/B/B/A 运行；A 为 deterministic auto，B 为
performance auto，每模式每长度 n=6。八轮实际 mode、开关及 tile IDs 回执已核对，
正式请求输入哈希和 submitted/accepted/consumed/received 帧数均一致。
输出间隔先对每个请求取 p95，再对六次取中位数；下表均排除冷启动、预热及 profile。

| SP / 输入 | A 中位耗时 | B 中位耗时 | 耗时减少 | FPS A→B | 首帧 A→B | 输出间隔 p95 A→B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SP1 / 10 s | 45.638920 s | 44.648375 s | 2.17% | 5.25867→5.37534 | 4.46172→4.31148 s | 3.27163→3.20303 s |
| SP1 / 60 s | 284.323751 s | 274.694724 s | 3.39% | 5.06481→5.24219 | 4.47141→4.33633 s | 3.29790→3.22566 s |
| SP2 / 10 s | 41.312177 s | 39.967447 s | 3.26% | 5.80945→6.00489 | 4.46976→4.38843 s | 2.86836→2.78729 s |
| SP2 / 60 s | 245.235349 s | 241.692463 s | 1.44% | 5.87195→5.95799 | 4.45711→4.40763 s | 2.89083→2.81497 s |

| SP / 输入 | A 耗时范围 | B 耗时范围 | 冻结服务门槛 |
| --- | ---: | ---: | --- |
| SP1 / 10 s | 45.4263–45.8337 s | 44.1279–45.9252 s | 通过 |
| SP1 / 60 s | 278.6675–289.5528 s | 271.8475–277.7043 s | 通过 |
| SP2 / 10 s | 40.1379–42.2626 s | 39.7292–40.6549 s | 通过 |
| SP2 / 60 s | 242.5223–248.2268 s | 239.5687–248.3457 s | 通过 |

冷请求指模型加载后的进程首个请求，进程间复用按 SP 区分的磁盘编译缓存，
不代表全新系统冷启动。零 GPU 的 trace 分析、测试准备和导出与部分服务测量
重叠且共享 CPFS，存储竞争未完全隔离；小幅端到端差异仅描述本轮固定 workload，
不据此声称稳健的普遍收益。SP2 长流 source-to-output p95 的最差样本从 A 15.131 s
增至 B 17.614 s；中位数改善不代表尾部全面改善。10/60 秒输入仍是 5 秒素材重放，
不是自然长流质量证据。

SP1/SP2 的 A、B 各独立生成原始 120 帧 / 24 FPS / 5 秒，单臂 832×480、无音轨，
保留六图、四步、seed42 和相同 TAEH3 decoder；未重放，也未截取长流。
早期零 GPU 导出 `dlc1s18v7uwrs3kh` / `dlcl28r2iny4b6m2` 均 Succeeded，
两组各有 A/B 独立视频及从原始 RGB 直接编码的并排版。最终归档再次验证输入和输出
原始字节哈希，与已交付的早期 5 秒视频来源相同。**人工画质仍待用户判断。**

### 全量媒体、逐卡显存与缓存

零 GPU 后处理作业 `dlc1mtn8y2rtsi65` 对 48 个正式计时、4 个 profile、4 个原始
5 秒请求输出逐一核验原始 RGB 大小与 SHA256、四类帧数及结束状态，完成 MP4 编码、
ffprobe 和完整解码；56/56 通过。两组并排视频另行核验。它证明媒体完整可解码，
不证明内部完整 DiT latent 数值 parity 或人工画质通过。

72 条正式请求/rank 显存记录按实际 session ID 对齐；下表取六次正式请求最大值，
加载快照按找到的 dispatch 数 / 总 dispatch 数列覆盖度，不用单卡代替其他 rank。

| SP / mode / rank / 输入 | allocated 峰值 GiB | reserved 峰值 GiB | EOS encoder bytes | 加载快照覆盖 |
| --- | ---: | ---: | ---: | ---: |
| SP1 / deterministic / 0 / 10 s | 229.413 | 229.941 | 0 | 96/96 |
| SP1 / deterministic / 0 / 60 s | 229.413 | 229.941 | 0 | 516/516 |
| SP1 / performance / 0 / 10 s | 229.413 | 229.961 | 0 | 96/96 |
| SP1 / performance / 0 / 60 s | 229.413 | 229.961 | 0 | 516/516 |
| SP2 / deterministic / 0 / 10 s | 179.188 | 179.783 | 0 | 96/96 |
| SP2 / deterministic / 0 / 60 s | 179.188 | 179.783 | 0 | 516/516 |
| SP2 / deterministic / 1 / 10 s | 179.226 | 179.705 | 0 | 96/96 |
| SP2 / deterministic / 1 / 60 s | 179.226 | 179.822 | 0 | 516/516 |
| SP2 / performance / 0 / 10 s | 179.188 | 180.154 | 0 | 96/96 |
| SP2 / performance / 0 / 60 s | 179.188 | 180.154 | 0 | 516/516 |
| SP2 / performance / 1 / 10 s | 179.226 | 180.154 | 0 | 96/96 |
| SP2 / performance / 1 / 60 s | 179.226 | 180.154 | 0 | 516/516 |

| SP / mode / rank | 加载 allocated 峰值 GiB | 加载 reserved 峰值 GiB |
| --- | ---: | ---: |
| SP1 / deterministic / 0 | 128.165 | 128.195 |
| SP1 / performance / 0 | 128.165 | 128.195 |
| SP2 / deterministic / 0 | 128.165 | 128.195 |
| SP2 / deterministic / 1 | 128.165 | 128.195 |
| SP2 / performance / 0 | 128.165 | 128.195 |
| SP2 / performance / 1 | 128.165 | 128.195 |

EOS 快照不是 dispose 的替代证据；encoder 独立生命周期测试另证实释放后 retained
bytes = 0。A/B 服务进程退出均出现单个 semaphore 的 resource_tracker 清理警告，
不能据此声明所有主机资源已证明无泄漏。每 rank host 阶段累计保留在 JSON 中，
不将异步 host 时间称为 CUDA kernel 时间，也不跨 rank 相加。

### Encoder trace 与 Perfetto

12 份原始逐 rank trace 已完整采集、离线分析、导出并回读校验 SHA256。
归因使用 CPU scope 嵌套及 correlation ID；各 rank 保持独立，不合并时间线。

| 配置 | 10 秒输入 CUDA kernel 累计时间 | 60 秒输入 CUDA kernel 累计时间 |
| --- | ---: | ---: |
| SP1 deterministic | 8.828 s | 51.502 s |
| SP1 performance | 8.211 s | 47.907 s |
| SP2 deterministic rank 0 / 1 | 4.426 / 4.420 s | 25.816 / 25.784 s |
| SP2 performance rank 0 / 1 | 4.116 / 4.113 s | 24.014 / 23.995 s |

这些是有 profiler 的 GPU 工作量累计值，不是无 profiler 的端到端耗时。
SP2 的 60 秒输入各 rank gather GPU 累计为 7.49–54.92 ms，CPU scope 累计为
67.09–73.24 ms；两者不能相加。未归因的 kernel 时间单独保留。

原始 JSON 最大约 6.79 GB。单卡 deterministic 10 秒原始文件在 Perfetto
实际打开；完整 60 秒原始文件的本机 UI 导入因资源压力中止，**不记为通过**。
随后按 [官方本地处理器方法](https://perfetto.dev/docs/visualization/large-traces)
尝试 Perfetto v58.2 的 loopback HTTP accelerator，也因本机内存及交换空间压力
在 UI 连接前停止；保留失败记录，没有据此改记完整原始文件通过。
另生成无时间裁切、无 rank 合并的浏览视图，保留全部 CUDA kernels、GPU memory
事件、H3/Zing CPU/GPU annotations 及 metadata，省略底层 CPU ops/runtime/flow。
**12/12 浏览视图均实际载入 Perfetto，目标事件数量与原始 trace 一致**；保留
`slice_spill_overlapping_complete_event` 导入诊断及截图。原始文件继续用于完整
CPU/CUDA 归因，精简视图不能代替完整 CPU 调用栈。performance 的 GPU gather
annotations 少于 CPU scopes 是原始记录的特点，不能据此推断漏采 kernel。

CPU 分析作业 `dlcry1rt6iy4cdg0`、浏览视图作业 `dlcd8x4i2vu9eq29` 均 Succeeded。
私有 OSS 证据位于本轮根目录的 `traces-early/`、`browser-views/`；各自 manifest
记录文件 SHA256，原始 trace inventory 同时记录未压缩文件。

SP1/SP2 A/B 的 **96 份服务原始 trace**（每模式/rank 16 份）已完整解析并导出。
CPU 作业 `dlc7zd9myvb9prrm` Succeeded；最终后处理逐文件重新核对原始 SHA256
后复用该完整解析结果，重复解析的 `dlcgx8c966y5nale` 已主动停止，原日志保留。
六个模式/rank 组合均实际在 Perfetto 打开代表原始文件和对应浏览视图。CPU ops、CUDA runtime/driver、全部 kernels、
CPU/GPU annotations 的类别数与原始文件一致。SP1 每份抽检有 88 条、SP2 有
106 条深层 `python_function` 被 UI 的嵌套深度上限省略，并保留 overlap 诊断。
所有抽检浏览视图的 kernel 和目标标记计数匹配；实际查看 CPU 主线程、CUDA stream
及 H3 tile 范围。这是六组代表文件 UI 抽检，**不是 96 份逐一 UI 验收**，也不是
完整 Python 调用栈无损展示。原始 JSON 不受 UI 限制影响，继续完整保留。

独立 profile 的耗时和正式计时分开保存在 `matrix.json` 与本地报告中；例如 SP2 A
对 240 帧采集 trace 耗时 1382.946 s，不能用它计算正常生成收益。

### 交付与剩余决策

私有证据根目录：
`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing195-20260928/`。
`final/export-manifest.json` 记录全部输出 SHA256，其本轮文件 SHA256 为
`f64ebf479aee81e1ae46f2662b950a9d3520454f1250f317e3319c436aa72f0a`。
`final/` 包含 56 份请求视频、6 份 5 秒展示视频、12 份 encoder
原始 trace、96 份服务原始 trace、统计/验收 JSON 和包含日志及数值 oracle 的
`metadata-final.tar.gz`；原始服务 RGB 仍保留在 CPFS 本轮目录。
`service-early/` 与 `browser-views/` 保留可浏览视图；`review-r2.tar.gz` 为附实际 UI
截图、统计与两组 5 秒媒体的私有评审包。完整大文件不强行加载到本机，也不公开发布。

本轮独立采集和技术验收完成。默认仍关闭；数值门槛失败、SP2 短流首 latent 尾部
未改善、人工画质待判断、SP4 未实测，以及同步 main 后组合默认策略未做本轮 GPU
重测，均继续明确保留。PR 需要评审与采用决策；没有自动合并。

以下保留 2026-09-24 首轮历史结果，其“未执行”状态只描述当时那一次作业。

---

# 2026-09-24 首轮结果

## 结论

**数值门槛失败，保持默认关闭。** 作业 `dlck7atqin5i9xli` 在 SP1 performance
第一例停止，未执行依赖的 SP2、正式计时、服务或 profile。没有可报告的正式提速，
也没有完整媒体或 Perfetto UI 验收。此结果不支持默认启用或无损声明。

测试前已在 [验收计划](PERFORMANCE_ENCODER_PLAN.md) 冻结双门槛：
`max_abs <= 0.25` 且 `relative_l2 <= 0.005`。这是工程诊断标准，不等于业务质量许可。
本次不调整该门槛。

## 冻结条件

- 测试源码：`a47f46f231a2f4837b469874689b74d7d6aaa550`；后续结果文档提交不改变受测代码。
- 同节点两卡，实际报告名称 NVIDIA L20D、计算能力 10.3、每卡 274,113 MiB。
  已执行阶段为 SP1，仅使用第一卡；两卡分片尚未验证。
- PyTorch `2.13.0+cu130`、CUDA `13.0`、cuDNN `92000`、NCCL `2.29.7`；
  cuDNN TF32 开启、matmul TF32 关闭、benchmark limit 10、matmul precision highest。
- 权重/配置共 69 个物理文件、163,163,082,223 字节；完整清单 SHA256：
  `129a38b0215796651b92da5e6ad48b3bfef5157515df024cc9263a8322326d1b`。
- source RGB SHA256：`d491562d9aae82ca0cd1d22db6d125ba06fb090c489faf37445187bfc21c7213`。
  120 帧真实素材重复构造长流，每 5 秒存在重放接缝，不是自然长视频质量证据。
- 六图资产已核验并冻结，但完整服务阶段未执行，不能称六图端到端通过。

## 实测数值

SP1 deterministic 完成 140 个 EOS/packetization 案例（1–34、120 帧 × 四种分包），
moments、indexed normalized posterior 与基准逐字节一致；未来帧不变性、交错会话、
reset、输出所有权、dispose，以及注入 kernel 异常后的恢复通过。

SP1 performance 第一例为 1 帧、分包 `[1]`。所有张量格式相同且 finite：

| 对比 | moments max abs | moments relative L2 | normalized latent max abs | normalized latent relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 同 performance，auto 对 replicated | 0 | 0 | 0 | 0 |
| performance 对冻结 deterministic SP1 | **0.5703125** | 0.002524650960877324 | 0.24658203125 | 0.004191065345596191 |

跨模式 moments 超出最大绝对误差门槛，normalized latent 在本例内通过诊断门槛但
不逐字节一致；不能把后者推广为全矩阵或成片质量通过。失败发生于 SP1 replicated
执行路径，不能归因于 tile 分片或 gather，也不能因此证明 SP2 performance 安全。

## 验收状态

| 项目 | 状态 |
| --- | --- |
| 默认关闭的独立开关、session 冻结及执行回执 | 已实现 |
| CPU 回归及验收器测试 | 656 项通过；真实 CPU/Gloo，不替代 GPU |
| GPU SP1 deterministic 数值/生命周期/异常恢复 | 通过 |
| GPU SP1 performance 跨模式数值 | **失败：第一例 moments max abs** |
| GPU SP2 分工、数值及生命周期 | pending：前置 gate 失败后未执行 |
| 独立进程 A/B B/A，10/60 秒正式计时 | pending：未执行；冷启动/预热不计入正式收益 |
| 每 rank profile、Perfetto UI、长流显存/缓存平台期 | pending：未采集 |
| 六图服务、完整媒体解码与视觉验收 | pending：未生成 |
| SP4 | 未实测，不宣称支持 |

## 私有证据

CPFS 根目录：`/cpfs/world-model/sglang/evals/zing195-20260924/gpu-r1`。
保留每阶段 report、完整日志、硬件/包环境、冻结 deterministic SP1 oracle、
performance 首帧 oracle，以及失败张量。`numerical-failure.pt` 中 reference 是同模式
replicated 张量；跨模式失败必须使用 `sp1-reference-deterministic/frames-1.pt`
对照，不能误读该文件为跨模式基准。

零 GPU 作业 `dlc1m7veve58bbkf` 将全部本轮结果及冻结清单导出为私有 OSS
`world-model/sglang/evals/zing195-20260924/gpu-r1-failure-evidence.tar.gz`，
bucket 为 `leap-worldmodel-thailand-single-az`。归档共 40,866,141 字节，SHA256
`7c0895c15e9c5b38ab9c6ab61e257a2f7f2d2f2335da121131ee0a28f4e0ce23`。逐文件 SHA256 见归档中的
`gpu-r1/evidence-index.json`。原始素材与媒体未发布到公开入口。

依赖阶段按预先的失败规则停止。任何后续诊断须使用新结果目录并保留本次失败，
不能放宽阈值后将其改记为通过；性能和质量采用仍需独立证据。
