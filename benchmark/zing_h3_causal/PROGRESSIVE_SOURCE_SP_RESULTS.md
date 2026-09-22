# Progressive source：SP1 / SP2 实测结果

## 当前六图 source.append（2026-09-22）

矩阵冻结 `386543ac257d7fd190707a1a1796ad6006a4873d`；原始 Picture1 PNG 加五张配对图，
无 prompt video。832×480、24 FPS 输入、ODE1000 四步、BF16 encoder、TAEH3 FP16。
各长度一预热三次无 profiler，下表均为中位数；localhost 收帧不代表公网或浏览器呈现。

| 路径 | 5 秒 FPS | 10 秒 FPS | 60 秒 FPS | 5 秒请求首画面 |
| --- | ---: | ---: | ---: | ---: |
| SP1 clip17 | 5.445 | 5.238 | 5.217 | 4.537 s |
| SP1 progressive | 5.540 | 5.216 | 5.036 | 4.418 s |
| SP2 clip17 | 5.851 | 5.761 | 5.955 | 4.559 s |
| SP2 progressive | 5.516 | 5.310 | 5.369 | 4.763 s |

独立单次同步首 latent：SP1 2.072→1.029 s；SP2 rank0 1.864→1.018 s、
rank1 1.864→1.014 s，真实依赖 17→1 帧。这不是首画面时间或三次中位数。
整作业每秒显存采样峰值 SP1 244942 MiB；SP2 195506/195404 MiB，包含加载、
预热、并发与 profile，不能当 encoder 独立峰值。encoder 状态峰值 3393245184 bytes；
60 秒 KV 平台期在 10 秒同版上下界内，累计计算 57444→335580，未见前缀重算增长。
60 秒输入的客户端待处理帧数最多 67（上限 68）；背压有界不代表实时。
按 24 FPS 采集时间线计算，渐进 SP1 source-to-output 的 p50/p95/max 三次中位数为
115.0/223.2/227.5 s，SP2 为 108.4/205.8/209.8 s；未跟上输入的延迟会持续累积。

SP1/SP2 的 624/631 件产物逐 SHA 回读，EOS1–34、分包、生命周期、隔离、cleanup
及四段视频完整解码复核通过。最新运行代码 `5d20e78e64e2d5f4e432dbb94edf663fb89c2b20`
另完成默认入口、独立 target-frame0 Video1+五图、双 rank 异常释放增量；87 件产物
逐 SHA 复核通过。target-video 三次 FPS 中位 5.589，分包输出一致；不代表 native 等价。
CPU 2028 项通过，不能替代上述 GPU 证据。独立双 rank 短段 profile 补证的
56 件产物逐 SHA 回读，10 份 trace 均解析到 CPU/CUDA 事件；不混入正式 FPS。

原始产物：`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing148-final-20260922/`
下 `sp1-r2`、`sp2-r2`、`sp2-incremental-r1`、`sp2-allrank-r1`；各目录含独立审阅回执。
[完整复核与测量口径](https://github.com/seedleap/sglang/pull/148#issuecomment-5778601514)。
旧/新最终 RGB 仍有有限差异（SP1 MAE 17.887、SP2 19.293，uint8）；不宣称逐位等价。
24 FPS、连续场景画质未通过，分别按 issue #131/#153 跟进。10/60 秒仅为技术拼接长测。

## 历史输入与旧冻结结果（保留，不代表当前六图验收）

全链路 MessagePack 输入、渐进 encoder、DiT、TAEH3、客户端收帧已跑通；
**当前配置未达到 24 FPS，不能作为实时交付通过，也不能将 PR #148 标为全部验收完成。**
用户允许 encoder 数值差异仅报告，因此保留原失败，不用 exact parity 阻塞本轮测速。

## 冻结与证据

- 源码：`67cacfe956e1480cb01fffd954ebe341f2ea1911`。
- 镜像 digest：`sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`。
- SP1：`dlc1bot34aihc04c`；SP2：`dlc1ypy70qr3d7ba`。均优先级 3，Succeeded，worker exit 0。
- 分别使用 1 / 2 GPU、16 CPU、128 GiB。实测 GPU 报告 NVIDIA L20D，
  CC 10.3，每卡 274113 MiB；Torch 2.13.0+cu130。
- ODE step1000、performance encode、BF16 encoder、TAEH3 fp16、832×480 / 24 FPS、
  四步 `[1000,750,500,250]`、seed 42、checkpoint ChunkPlan `[1,2,2]`、split_first=false。
- 两组资产哈希一致；profile SHA256 均为
  `fa64a9fee515a479573ac5b10009aa1f6ee94acf8334007173598c2a6da712c4`。
- 完整模型/decoder/input 哈希见各运行的 `asset-hashes.json`、`checkpoint-receipt.json`；
  实际配置见 `profile.json`、`model-config.json`、`decoder-config.json`、各模式 `request.json`。
- OSS 原始证据根目录：
  `oss://leap-worldmodel-thailand/world-model/sglang/evals/zing148-20260922/service-sp1-r1/`
  与同级 `service-sp2-r1/`。CPFS 对应 `/cpfs/world-model/sglang/evals/zing148-20260922/`。

## 客户端实测速度

每组一次 10 秒预热后，三次 10 秒无 profiler 测速。60 秒 throughput/paced 各一次，
不是三次重复。真实 MessagePack 分包模式 `[1,3,17,2,5]`。
FPS = 客户端实际收到帧数 / 从请求开始到完成的时间；包含首帧等待，
服务与客户端在同一 worker，通过 localhost WebSocket，不代表公网或浏览器呈现。

| 指标 | SP1 | SP2 |
| --- | ---: | ---: |
| 10 秒三次 FPS | 3.581 / 3.561 / 3.518 | 4.677 / 4.660 / 4.700 |
| 10 秒 FPS 中位数 | 3.561 | 4.677 |
| 10 秒首帧中位数 | 2.773 s | 3.093 s |
| 60 秒 throughput：1440 帧耗时 | 452.569 s | 329.018 s |
| 60 秒 throughput FPS | 3.182 | 4.377 |
| 60 秒 paced FPS | 3.178 | 4.397 |
| 60 秒 paced 模拟播放累计停顿 | 391.206 s | 265.453 s |

停顿由客户端以 0.5 秒启动缓冲、24 FPS 播放时间线估算，并非真实播放器测量。
所有 14 个会话（含两次预热）的 submitted/accepted/consumed/received 均分别为
240 或 1440，均有一次 `session.completed`，alignment 由客户端逐包校验。
每个 SP 的三次 10 秒输出 SHA256 一致；SP1 与 SP2 的输出 SHA256 不同，
不据此声称跨 SP 数值等价。

双卡相对单卡：10 秒中位数 **1.314×**，60 秒 **1.376×**；并非线性扩展。
双卡 60 秒吞吐只达到 24 FPS 的 **18.2%**，仍需约 **5.48×** 提速。
双卡首帧没有改善。冷预热首帧 SP1 71.440 s、SP2 115.103 s，单独保留，不混入正式结果。

## 阶段、状态与释放

`server.log` 真实 source progress 包含 `latent_submitted`、`dit_started` 和 dependency
水位。本轮 `measure_cuda=false`，这些是 host 提交事件，不能称为同步 `latent_ready`。

| 60 秒 throughput，rank 0 | SP1 | SP2 |
| --- | ---: | ---: |
| encoder host scope 累计 | 50.804 s | 51.289 s |
| DiT host scope 累计 | 395.638 s | 267.939 s |
| decoder host scope 累计 | 0.547 s | 0.566 s |

以上是未做逐阶段 CUDA 同步的主机侧 scope，不能当作 GPU kernel 时间、GPU 占比，
也不能直接用来计算某个优化的理论收益。当前证据指向 DiT 为优先排查方向；
SP2 降低了 DiT scope，但 replicate encoder 没有获得相应缩短。
需要独立 CUDA trace 才能区分算子、通信、同步等待和主机调度；本轮无此 trace。

- 10/60 秒的每 rank encoder 保留状态峰值均 **3393245184 bytes（约 3.16 GiB）**。
- KV temporal chunks 上限均 21；KV 峰值 SP1 **39169474928 bytes**，
  SP2 每 rank **19585065328 bytes**。
- 每个会话最后一条 ended 记录的 encoder retained bytes、KV bytes、bridge bytes 均为 0；
  正常 EOS 的 ready/decoder/pending 状态也清空。模型常驻和 allocator reserved memory
  不会随会话归零；该结果不能替代取消/异常释放验收。
- encoder `temporal_positions` 是累计计算计数，不是缓存长度：10 秒 57444，
  60 秒 335580，约 5.842×；每帧约 239.35 / 233.04。支持本轮长流计算随帧数近线性，
  不等同于完整复杂度证明或旧路径 A/B。
- 全作业 1 秒 nvidia-smi 采样峰值：SP1 GPU0 **178950 MiB**；
  SP2 GPU0/GPU1 **164674 / 164694 MiB**。不是瞬时峰值，也不是可部署显存下限。
- 60 秒 throughput 的 PyTorch peak allocated：SP1 rank0 **184313430016 bytes**；
  SP2 rank0/rank1 **164745795072 / 164786903552 bytes**。
  `GPUWorker` 在每次 forward dispatch 开始时重置此计数；这里汇总该请求各
  source-step 快照的高水位，不包含此前会话的历史峰值。它不是 encoder 单独峰值，
  也不覆盖 source-step 记录之后的所有临时分配；整作业显存另见 nvidia-smi 采样。

## 验收边界与后续工作

本轮完成的是用户优先要求的全链路贯通与当前 SP1/SP2 速度分析。
24 FPS 未通过，画质未验收；distinct-clips 拼接输入仅为技术长流，不能证明连续场景画质。
encoder 诊断 r2 的 future perturbation 首 1/3 latent exact 通过，但对旧 clip17 仍有
max_abs=0.125、RRMSE=0.002116739 的原零容差失败；未将其改标通过。

PR #148 全量交付仍需：针对上述瓶颈优化后同冻结语义重测；完整 EOS1–34 / 分包与旧17
A/B；真实暂停恢复、慢消费者、取消/end竞态、重连、隔离及异常释放；连续素材画质。
CPU 测试和正常 EOS 结果不能替代这些测试。issue #131 的整链路 24 FPS 与画质仍未完成。
不自动合并 PR，不关闭 issue，不重启已停止 SP8，也不对未测试更高 SP 做线性外推。
