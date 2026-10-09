# SP2 在线 source 编码与联合 DiT 补测

## 为什么要补测

2026-10-08 用户再次明确：后续业务性能与验收默认走在线模式。此前 VAE encoder
和专用 attention kernel 的最近几轮使用离线 generate，没有落实这个要求。
离线 kernel 的 API -1.00% / DiT -2.26% 不能称为在线收益。

旧 SP2 在线测试测得 joint off / cross0 全程中位数 32.9279 / 31.0218 秒
（下降 5.79%）；它不是独立 DiT GPU 计时，也没有覆盖新专用 kernel。
历史证据见 [JOINT_FORWARD.md](JOINT_FORWARD.md)。

## 在线和离线的差别

离线完整 source 已知，可先编码整段，再安排跨块联合；在线 source 逐帧到达，
只能消费已收到的帧，并维护增量卷积状态、输入背压和输出交付。渐进编码按一个
17 帧 clip 内的 1/8/8 输入推进，产生 1/2/2 个 latent；不能为了加速偷看未来帧。
跨块合并受下一块是否就绪制约，不能直接用离线三分支次数推算在线收益。

当前两个入口的 source 预处理也不同：本案例离线 source/target 每 latent
分别为 880/390 token，在线两者均为 390。因此不是仅换了传输方式的同形状测试。

窗口限制历史块数，当前 Q 也有界。“长 query”是相对单 token 解码而言：
832×480 经 VAE 16 倍空间压缩为 52×30，再按 2×2 patch 得到每 latent
26×15=390 token；当前 1/2 latent 块为 390/780 Q。三分支可能为
780+780+780=2340 Q，绝不是全视频 Q 无限累积。历史 KV 还含固定参考前缀，
且每个历史块含空间 token；window=21 不是 21 个 KV token。SP2 Ulysses
交换后每卡处理完整序列、部分 heads，不能把 attention 的 Q/K 长度再除以 2。
实际 Q/K/H 形状由本次逐 rank 回执记录，不用公式替代实测。

## 本轮配置

同一双卡作业，原始 source + 六图、120 帧、832×480、24 FPS paced WebSocket，
progressive encoder、SP2、resident DiT、FA4、四步 `[1000,750,500,250]`、seed42、
CFG=1、window21/sink3、joint wait=0；native stable decoder，无音频。
每档独立进程，一次完整预热、三次正式请求、一次独立 CUDA-event 诊断。

| 档位 | encoder mode | source tile 分工 | joint |
|---|---|---|---|
| perf-tile-off | performance | SP2 | off |
| det-tile-cross | deterministic | SP2 | cross_chunk |
| perf-replicate-cross | performance | 每卡重复 | cross_chunk |
| perf-tile-cross | performance | SP2 | cross_chunk |

启用现有在线实现，不将离线 `SGLANG_ZING_OFFLINE_ENCODER_PARALLEL` 冒充在线开关：

```sh
export MINIMAX_H3_PROGRESSIVE_ENCODER_TILING=auto
export MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO=1
export MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE=1
```

模型 `zing_config.json` 的 `vae_config.encode_mode=performance` 会在编码范围启用
`cudnn.benchmark=True`、`cudnn.deterministic=False`、deterministic algorithms=False，
退出后恢复。profile 的配置哈希同步更新。对照组显式设 `TILING=replicated`。
验收读取每个 pop 的 `local_tile_ids`、gather、fallback、实际后端策略，而非仅看配置。

## 计时与数值口径

正式请求只记录客户端总时延、TTFF、交付间隔以及已有 host scope；不插入 CUDA
事件或同步。独立诊断在现有 DiT/decoder scope、encoder pop、prepare 范围记录
CUDA event，每次 dispatch 结束读取。它是 stream 时间区间，含等待与发射间隙，
不是纯 kernel 累计时间；按 rank 分开，不跨卡相加，不混入正式端到端收益。
prepare 包含初始化工作，不能将其与内部子阶段重复相加。

performance 的历史数值差异保留，本轮速度测试不等于无损准入。保留逐次 RGB、
重复性与跨组误差、素材和模型回执；不调整旧阈值，不将无音轨视频当音视频验收。

## 执行状态

源码基座 `1224489aea6923b47a5ab01a74b87f65f3fad04b`，诊断工具单独冻结。
CPU 准备作业 `dlcdnts7efy1u5vn` 成功；首次准备 `dlc15ymyoayuuid7` 因 CPU 镜像
没有 ffmpeg 失败，保留回执，修正后使用独立目录。GPU 作业 `dlc1i66i2y3gf7zd`
已完成并通过结果完整性核验，见下方 r1 数据；三档 joint 补测单列，不覆盖旧样本。

## 首轮在线结果（r1，已核验）

GPU `dlc1i66i2y3gf7zd` 与 CPU 后处理 `dlc9jbmruabsraxi` 均成功，worker exit=0、
COMPLETE、全部四档各一次预热/三次正式/一次独立诊断齐全。12 份正式 RGB 均为
120 帧，MP4 完整解码通过。以下不包括专用 attention kernel。

| 配置 | 正式全程中位数/秒 | TTFF/秒 | source host/秒 | DiT host/秒 | decoder host/秒 |
|---|---:|---:|---:|---:|---:|
| performance + tile SP2 + off | 33.2131 | 7.0384 | 2.3002 | 12.0592 | 17.1401 |
| deterministic + tile SP2 + cross | 31.4651 | 6.5847 | 2.4655 | 10.0233 | 17.2120 |
| performance + replicated + cross | 32.9734 | 7.0646 | 3.8616 | 10.1463 | 17.2244 |
| performance + tile SP2 + cross | 31.8785 | 6.6406 | 2.2701 | 10.2065 | 17.2206 |

host 列为 rank0 三次正式累计 scope 的中位数，异步提交时间不等于纯 GPU 计算。
各阶段中位数不能精确加成客户端中位数。prepare 约 0.97–1.01 秒，包含 Qwen
约 0.58 秒，二者不能重复相加。

独立诊断（每档仅一次）的 rank0 CUDA 时间区间如下；包含流内等待与发射间隙，
不混入正式延迟或作为稳定加速比：

| 配置 | prepare/秒 | encoder pop/秒 | DiT/秒 | decoder/秒 |
|---|---:|---:|---:|---:|
| performance + tile SP2 + off | 0.9570 | 1.9683 | 11.9086 | 17.0663 |
| deterministic + tile SP2 + cross | 0.9936 | 2.3389 | 10.3418 | 17.0519 |
| performance + replicated + cross | 0.9592 | 3.5893 | 10.0863 | 17.0568 |
| performance + tile SP2 + cross | 0.9751 | 1.9632 | 10.5407 | 17.0570 |

- 实际分工通过：12 个 tile，rank0 为 0/2/4/6/8/10，rank1 为 1/3/5/7/9/11；
  SP2 每次 gather 一次。replicated 两卡都处理 0–11。各 pop 的 encode_mode、
  cudnn benchmark、deterministic 和 fallback 回执均核验。
- performance 下 source host 中位数 3.8616→2.2701 秒（约 -41.21%）；独立
  encoder pop CUDA 区间 3.5893→1.9632 秒（约 -45.30%）。两者计时范围不同。
- cross 每个正式请求实际 104 次 forward：prefix1、独立 denoise66、final15、
  source_first15、三分支7。15 次 source_not_ready 回退。在线并没有每块都达到
  三分支合并；不能套用离线 20 次三分支的收益。
- 两卡记录的联合形状为 `390+390`、`780+780`、`780+780+780`。此前专用 kernel
  白名单未覆盖这些形状，不声称该 kernel 已启用于本次在线测试。

### 波动与数值限制

performance+tile+cross 三次全程为 31.3988 / 35.2609 / 31.8785 秒，spread
约 12.11%，超过参考 10% 稳定性门槛，全部保留。慢样本没有对应的同量级 DiT
host 增长，但现有证据不足以归因到网络、调度或其他具体等待。正式中位数相比
off 下降约 4.02%，不能据此宣称稳定的在线加速。

各组内部三次 RGB 均逐字节一致，但跨组不一致：perf tile cross 相对 perf tile
 off 的 RGB8 MAE=15.7214、relative L2=0.15424、max_abs=228；det tile cross
与 perf replicate cross 相对该 off 的 relative L2 分别为 0.16518 / 0.17267。
这里比较的是客户端 RGB8，不是未量化 latent 的旧阈值。**本轮不能宣称 joint 在
performance 组合下无损。** 尚未隔离条件编码、source 编码与联合 DiT 各自的贡献，
不能直接把差异归因于其中一项。

为分别测 source_first/cross_chunk，并补齐差异定位，追加同一作业内的三档
performance+tile SP2 对照（off/source_first/cross），每档三次正式和三次独立
阶段诊断；预热记录条件/编码输出哈希，正式不做张量拷贝。补测作业
`dlc1nhby8aj6vct4` 已成功完成，结果见下方 r2；不替换本轮波动样本。

原始结果包 SHA256：`bbd4899b4f69f6bb6146477e80b17d7613243d4913cd96ae355da496ce4c0417`。
私有 OSS `world-model/results/zing-online-20261008-r1/`，含 summary.json、
12 份视频、完整日志、逐卡阶段/tile/shape 回执。结构化摘要见
[online_sp2_retest_r1.json](online_sp2_retest_r1.json)。


## 最终三档在线对照（r2，已完成）

GPU `dlc1nhby8aj6vct4`、零 GPU 后处理 `dlc138ff9mv8fwdq` 均 Succeeded，
worker exit=0。三档均为 performance + source tile SP2；每档独立服务、一次
完整预热、三次正式请求、三次独立阶段诊断。9 份正式结果均收到 120 帧，
结束事件、源素材哈希与完整视频解码检查通过。保持四步采样，未启用专用 kernel。

| joint | 正式全程中位数/秒 | TTFF/秒 | DiT host/秒 | DiT CUDA 区间/秒 | CUDA 区间下降 |
|---|---:|---:|---:|---:|---:|
| off | 33.3438 | 7.1523 | 12.1975 | 12.5552 | — |
| source_first | 31.8049 | 6.6990 | 10.5121 | 10.7738 | 14.19% |
| cross_chunk | 31.3082 | 6.6269 | 10.0733 | 10.3058 | 17.92% |

DiT host 分别下降 13.82% / 17.41%，全程分别下降 4.62% / 6.10%。CUDA 列
取 rank0 三次独立诊断中位数，包含 stream 等待/发射间隙，不是 kernel 时间之和。
rank1 DiT 区间依次为 12.5393 / 10.7645 / 10.2809 秒，不能两卡相加。
这些是本次有限重复的实测，不是所有在线输入的收益保证。

正式全程原始样本（秒）：

- off：33.3438 / 33.1699 / 33.8667。
- source_first：34.3309 / 31.6290 / 31.8049。
- cross_chunk：31.2479 / 31.4091 / 31.3082。

### SP2 performance 阶段分解

cross_chunk 的 rank0 正式 host 中位数：prepare 0.9804 秒（内含 Qwen
0.5908 秒）、source encoder 2.2401 秒、bridge 0.0116 秒、DiT 10.0733 秒、
decoder 17.2079 秒；客户端全程 31.3082 秒。不同 scope 的中位数不严格可加。
独立 CUDA 区间依次为 prepare 0.9766、encoder pop 1.9713、bridge 0.0111、
DiT 10.3058、decoder 17.0605 秒。encoder pop 与 source host 范围不同。
prefix forward 在 step/drain 的 DiT scope 内；不能把它说成 prepare 的子阶段。

当前最大单项是 native stable decoder，约 17.2 秒。source 多卡 tile + benchmark
已经实际启用并核对了 12 个 tile 的 6/6 分工；其相对重复编码的收益见 r1，
不能将 r1 的百分比与 r2 joint 百分比直接相加。

### 数值结论：性能测试完成，performance 无损准入未通过

每档内部三次正式 RGB 完全一致；跨档不同。source_first / cross_chunk 相对
 off 的 RGB8 MAE 为 16.1869 / 18.5842，relative L2 为 0.15880 / 0.16995，
max_abs 均 227。视频可解码不代表内容正确或业务效果合格。

仅预热记录的输入侧哈希提供了更具体的定位：

- rank0 的 Qwen context 与 row_role 三档相同。
- source_first 相对 off，六张 reference latent 中五张不同；cross 相对 off 六张相同。
- source_first 与 cross 相对 off，22 次 source encoder moments 及采样后的
  source latent 均不同：差异在进入联合 DiT 前已经存在。
- 同一档两卡 source moments 与采样 latent 相同，但五张 reference latent
  的两卡哈希不同；全局参考图和 Qwen 条件哈希相同。

这排除了“所有差异都首次发生在联合 DiT”这一解释，但**不能证明 joint 无误，
也不能仅凭哈希把原因锁定为 cuDNN benchmark**。独立进程的算法选择与编码路径
仍需用冻结同一组条件/source latent 的回放进一步隔离。哈希仅在预热采集，不能
声称正式请求也已逐张量比对。保留默认 off；不把此次 performance 组合当作
原先确定性 joint 验收的替代，不宣布该组合无损或默认启用准入。

### 可复核证据

源码基座 `1224489aea6923b47a5ab01a74b87f65f3fad04b`；源码包 SHA256
`82fe6f7d4a4b5694cb60eabb1cc2bbd48ce4047a70aab36b399fb0b9cebed47c`。
实际诊断工具包 tools-r4 SHA256
`fc4aaca94c337d43da08d046095f4b00fd1ded1c9ac62e001251fd08b02f1a95`。
结果包 SHA256
`d887a65859bd8b33c871565a8e0eafb82ca4956fe1400d8a3845f5b36f6847a0`，
下载后再次校验通过。两卡均报告 L20D、CC 10.3、274113 MiB；不据产品名推断架构。

结构化摘要：[online_sp2_retest_r2.json](online_sp2_retest_r2.json)。私有 bucket
`leap-worldmodel-thailand-single-az` 下
`world-model/results/zing-online-20261008-r2/` 包含全部日志、逐卡回执、9 份视频、
summary 与结果归档；未发布公网素材。所有本轮 GPU/后处理作业均已结束。
