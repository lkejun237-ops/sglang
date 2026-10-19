# TAEH3 阶段验收记录

> 历史记录：TAEH3 后端、配置及专用工具已移除。本文只描述下列冻结版本，复现需检出相应 Git 提交；当前部署使用 Lynn，不能沿用此处的质量或性能结论。

实现、数值正确性和功能性流式检查已完成；持续 24 FPS 与人工画质验收尚未通过，
因此本 PR 继续保持 Draft，不应据此宣称实时达标或可合并。

## 固定条件

- main 基线：`aa5474d37ef0c9b0b233dc156ca16d1d20aefc8b`。
- decoder gate 代码：`8b952ec4e69a1b184cb61b5d96be85fad4fa3502`。
- 上游 TAEHV：`011dfc2112197741c540e0bdd5b7b67bcc930771`；MIT license 随代码保留。
- `taeh3.pth` SHA256：`af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba`。
- 官方镜像：`lmsysorg/sglang:v0.5.19-cu130`，使用已核实的 ACR 镜像
  `zing-thailand-registry-vpc.ap-southeast-7.cr.aliyuncs.com/zing/sglang@sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`。
- 实测运行时：Torch `2.13.0+cu130`、CUDA `13.0`；设备报告 `NVIDIA L20D`、CC `10.3`、显存 `274113 MiB`。硬件判断按实测值，不按名称推断。
- 2026-09-21 用户更新：后续完整生成默认 **832×480、24 FPS、四步 `[1000,750,500,250]`、guidance 1、seed 42**。保留 240 帧 / 10 秒及 1440 帧 / 60 秒两项验收，不与旧 896×512 数字混作同规格对比。

## 已完成：真实 latent decoder gate

复用旧请求存档中的同一真实 DiT latent：`[1,24,72,32,56]`，对应 896×512 / 240 帧。
这是已有 latent 的独立正确性小试，没有重新生成旧分辨率请求。
原始 latent 字节 SHA256：`b953eafc2c7acc6d3a61b409825c6c3eaffdfaa269ced5a401a7bc2fd91a41ef`。

| 精度 | 预先规定的 allclose 容差 | 与固定上游最大绝对误差 | RMSE | warm decode 中位数 |
| --- | --- | --- | --- | --- |
| FP32（IEEE convolution） | atol 2e-5，rtol 2e-4 | 2.59876e-5 | 8.39493e-7 | 0.356417 s |
| FP16 | atol 2e-3，rtol 2e-3 | 0 | 0 | 0.071260 s |
| BF16 | atol 2e-2，rtol 2e-2 | 0 | 0 | 0.071023 s |

allclose 使用 `abs(actual-reference) <= atol + rtol*abs(reference)`，不是只比较最大误差和 atol。
三次独立 unprofiled warm FP16 wall 时间为 `0.071260232 / 0.071145061 / 0.071490963 s`，
对应 CUDA event 为 `0.071205116 / 0.071123390 / 0.071463966 s`。
这些是已有 GPU latent 的 decoder-only 时间，不含 DiT、输入准备、CPU 搬运、媒体编码或请求启动。
显存峰值 FP32 `7818.82 MiB`、FP16/BF16 `2453.20 MiB`，包含验证器保留的输出，不能直接等同于生产峰值。

`[5]`、`[1,2,2]`、`[3,7,2]` 输入分块，拼接结果均与 adapter 离线输出逐像素一致；
帧数均为 240。CPU 测试另覆盖单帧、首尾短片、跨组边界、逐 latent 只解码一次、batch/session 隔离、取消/reset 及精度设置恢复。
这里的分块覆盖不等于已验证 main 不支持的 list-valued AR 配置。

首次 FP32 检查未通过：默认 TF32 下，不同时间批大小产生最大 `0.0100867` 的差异。
修复为 decoder 内局部禁用 TF32，并对上游使用相同 IEEE 精度；**未放宽容差**。
该上下文退出后恢复调用方 flags，不改变 DiT 的精度设置。

历史 native 像素对照的 PSNR 约 `20.68 dB`，仅为数值描述，**不是画质通过**。
新 main / 480p 的 SP1/2/4/8 同 latent native 对照与四组代表场景均已完成。

## 作业与自动检查

PAI workspace `384`，region `ap-southeast-7`，quota `quota1betfyi1ilu`；本阶段峰值 8 GPU。

| 作业 | 结果 |
| --- | --- |
| `dlc16mtq8ae40ybs` | 零 GPU 预检失败；默认 sh 不支持 pipefail，已改为显式 Bash |
| `dlc9m7g963vro0vx` | 零 GPU 挂载、包与权重哈希、真实 latent 预检成功 |
| `dlc1p83xfu0w5wjb` | GPU 测试入口循环导入，decoder 未执行；已改测试启动顺序 |
| `dlcg00x6yp3k73lj` | FP16/BF16 gate 通过，FP32 TF32 gate 失败；保留原始证据 |
| `dlc1cgqtjejt2gkn` | 三种精度 gate 均通过；官方镜像内 228 项相关回归通过；Succeeded |

完整相关回归通过正常 `DiffGenerator` 入口初始化后运行 pytest，覆盖 Zing model、registration、media、realtime adapter 和 TAEH3。
PR 的独立 CPU CI 另运行 model-only 合约测试，避免引入 CUDA runtime conftest。

证据根目录：`/cpfs/world-model/sglang/evals/zing-h3-dmd/taeh3-main-20260921-r1`。
`decoder-attempt2` 保留失败数据；`decoder-attempt3/report.json`、`unit-tests.log`、`validation.log`、`job-exit-code.txt` 为通过记录。
本地副本：`/Users/chenshengdong/Downloads/zing-taeh3-main-20260921-r1`。
提交 manifest、脚本及每次回执保存在该本地证据目录；未把权重、原始视频或大型张量提交 Git。

## 验收结论

- TAEH3 decoder streaming、任意输入分块、session 隔离、取消/reset、FP32/FP16/BF16
  数值 gate 与 SP1/2/4/8 重复性通过。
- 832×480 的 warm 完整请求、四场景数值对照、60 秒 scheduler 流式、真实 WebSocket、
  媒体完整解码、trace 与累计 GPU 时间已测。
- SP4 与 SP8 的 scheduler / WebSocket 持续生成均未达到 24 FPS；SP8 只将 warm scheduler
  提升到 16.132 FPS、WebSocket 提升到 15.644 FPS。
- 四场景同步播放器已发布，PSNR/SSIM 与时间稳定性已记录；LPIPS 依赖不可用，人工画质
  尚未签收。持续 24 FPS 与人工画质签收是当前合并阻断项。
- TAE 权重和计算在 SP ranks 复制，不宣称 SP 能线性加速 TAE。进一步工作应先优化随历史长度
  增长的 DiT / scheduler 路径，再按同一 60 秒规格复测 SP8。

未单独叠加未进入 main 的 #117/#127；#120/#121 随后已通过更新 main 纳入。本 PR 未合并。

## 2026-09-21 后续执行

- 分支已合入 main `f1428c6d86`，包括 main 已接收的 #120/#121；没有单独合并未进入 main 的优化 PR。
- 新矩阵冻结代码 `7bf81580945f795e3614c2e9b7954e9dc3a55023`，新增逐 chunk wall/CUDA 计时；SP1/2/4 在同一四卡作业中串行执行。
- 作业 `dlcagohsa5b4yoqb` 已完成；624 项相关集成测试、480p 三精度 gate、分块一致性、SP1/2/4 性能、画质与持续流式矩阵均已取得有效证据。SP8 结果另见后文。
- 更新后本地 decoder 合约 48 项通过；PR CPU CI 通过，GitHub 显示 MERGEABLE。WebSocket SP4 真机功能检查已通过，持续 24 FPS 门槛失败，详见下表。
- 全部 PR 文件 pre-commit 检查通过。Mint 全站 validate / broken-links 被 main 已有的 MinWM 计划页第 599 行解析错误阻挡；本次修改的两页在隔离配置下 build validate 通过。

### 旧 main 上的 480p SP1 实测

以下数据来自 `8b952ec4e69a1b184cb61b5d96be85fad4fa3502`，与更新后的 main 矩阵分开保留。

| decoder | 三次 API 秒数 | API 中位数 | 视频 decode 中位数 |
| --- | --- | --- | --- |
| native | 56.055511 / 56.155899 / 56.214535 | 56.155899 s | 34.109501 s |
| TAEH3 FP16 | 见原始 sglang.json | 20.930914 s | 0.067054 s |

同一对输出 latent/audio 完全一致；PSNR 19.823865 dB、SSIM 0.578257，人工画质待确认。
8 个正式/profile MP4 均经 ffprobe 和完整解码检查：832×480、240 帧、24 FPS。
原 GPU 作业 `dlc13l57uyirs6gy` 计算已完成（CPFS computation exit code 0），但 OSS 挂载导出阻塞，最终达到 45 分钟时限而 Stopped。
零 GPU 作业 `dlc7yqqrunuwz3vl` 分片重新导出成功；81 个分片逐一 SHA256 校验通过。
原始 trace 仍保存在 CPFS；本次恢复导出包含媒体和指标，尚未导出 trace。

### 480p 逐 chunk 实测（新 main）

来自 `e2e-multi-r2/decoder-480p/report.json`，同一真实 480p latent，三次独立 warm stream；时间为同步 wall 毫秒。
首组输入五个 latent、输出五帧；普通完整组输出十七帧；结束补齐单独计时。
稳态仅统计三次运行中完整五个 latent / 十七帧的 39 次调用，排除尾部不足五个 latent 的缓冲调用；p95 使用 nearest-rank。完整逐次输入/输出计数保存在原始报告。

| 精度 | 首组 p50 | 稳态 p50 | 稳态 p95 | finish p50 | 整段 240 帧中位数 |
| --- | --- | --- | --- | --- | --- |
| FP32 | 21.428 | 21.546 | 21.572 | 13.622 | 313.274 |
| FP16 | 4.577 | 4.662 | 4.677 | 2.898 | 66.376 |
| BF16 | 4.545 | 4.644 | 4.670 | 2.874 | 66.030 |

这是 decoder-only，不含 DiT、输入准备、CPU 传输、WebSocket 和浏览器。

### 新 main 完整请求与 WebSocket 阶段结果

以下完整请求均为 240 帧，至少一次预热和三次无 profiler 测量。
API 时间包含媒体导出；不能等同于服务首帧或持续生成吞吐。

| SP | native API 中位数 | TAEH3 FP16 API 中位数 | 当前数值检查 |
| --- | --- | --- | --- |
| 1 | 56.343932 s | 21.271313 s | 三次 native/TAE 对照与重复请求均严格一致 |
| 2 | 34.089492 s | 15.454222 s | 三次 native/TAE 对照与重复请求均严格一致 |
| 4 | 22.653597 s | 12.481425 s | 三次 native/TAE 对照与重复请求均严格一致 |
| 8 | 18.463952 s | 11.580645 s | 三次 native/TAE 对照与重复请求均严格一致 |

真实 FastAPI / WebSocket 作业 `dlcgapssqcgggitc` 已 Succeeded，退出码 0。
运行时基于 `7bf81580945f795e3614c2e9b7954e9dc3a55023`，测量工具 overlay 为 `14e628eff2`。
四卡硬件门槛通过；流式测试显式使用 video-only 请求，原尺寸 raw RGB，经本机 WebSocket 传输，保留默认压缩设置。
完整 API 对照使用 video_audio；两种链路的耗时不能直接当作相同工作量比较。
不包含 WAN、浏览器渲染或模型加载时间；冷请求包含首次编译。
零 GPU 作业 `dlcnshn9mw1a76gj` 已完成证据导出，分片 SHA256 校验通过。

| 场景 | 帧数 | 首帧 | 总耗时 | 功能 / 性能结论 |
| --- | --- | --- | --- | --- |
| 冷请求 | 240 | 71.157428 s | 81.106247 s | 完整帧数与 payload 检查通过 |
| warm 持续 60 秒 | 1440 | 1.950669 s | 122.782155 s | 11.728089 FPS，累计卡顿 61.519408 s，**24 FPS 门槛失败** |
| 取消 | 16 | 1.903889 s | — | 第一批后取消成功 |
| 取消后重连 | 22 | 1.812233 s | 1.957410 s | 完整帧数通过 |
| prompt 更新 | 85 | 1.801930 s | 4.684194 s | event 123 provenance 与单调性通过 |

数据位于 `websocket-r4-sp4/websocket/websocket-report.json`，含逐批到达记录。
TAEH3 单块约 4.66 ms 并未消除完整链路的瓶颈。

SP8 同规格 warm scheduler 为 16.132321 FPS，首帧 1.369794 秒，总耗时 89.261799 秒，
chunk p50/p95 为 1.022606 / 1.520293 秒，累计卡顿 29.098696 秒。
SP8 WebSocket 为 15.643784 FPS，首帧 2.163891 秒，总耗时 92.049342 秒，累计卡顿
30.572704 秒。取消、取消后重连和 prompt 更新功能均通过。两条链路的持续 24 FPS 门槛均失败。

### 四场景数值对照

四组均已完成 native / TAEH3 FP16 生成，DiT latent 严格一致，输出均为 `[1,3,240,480,832]`。
逐帧像素 finite 且处于 `[0,1]`；这里为 video-only，audio 均为空，不能声称四场景音频已验收。
未给 PSNR/SSIM 事后设定通过阈值；主观画质等待视频审阅。

| 场景 | PSNR (dB) | SSIM | 时间差分 MAE |
| --- | --- | --- | --- |
| 文字 / 细节 | 21.566374 | 0.621504 | 0.042585 |
| 快速运动 | 22.104096 | 0.447728 | 0.064746 |
| 人物 | 32.900431 | 0.914392 | 0.012004 |
| 镜头切换 | 28.222029 | 0.803822 | 0.020897 |

镜像未安装 `lpips`，LPIPS 未测。原始每帧 PSNR/SSIM 与 temporal delta 保存在各场景 comparison.json。
SP4 WebSocket 长会话按完整 chunk 接收完成时刻计算，间隔 p50/p95 为 1.405899 / 2.333044 秒；包含网络传输和流水调度，不是 decoder 耗时。
整项 WebSocket 作业每秒采样的四卡显存峰值为 199350 / 199410 / 199710 / 199710 MiB；是设备占用采样值，不是精确瞬时峰值或逐 rank allocator 统计。

### 完整重复性与媒体复核

SP1/2/4 每种 decoder 三次正式 raw（共 18 份）均已按请求独立推导的 shape 检查：
latent `[1,24,72,30,52]`、audio `[2,320000]`，全部 finite。
九对 native / TAEH3 的 latent/audio 与同配置三次重复均逐元素一致；`sp124-repeat-parity.json` 为通过回执。
新 main 矩阵的 40 个正式/profile MP4 均通过 ffprobe 和完整解码：832×480、240 帧、24 FPS、10 秒；完整请求音轨为双声道 32 kHz，四组画质请求无音轨。
16 个 `faststart` 预览采用 stream copy，与原文件的音视频编码 packet 校验一致。

SP8 作业 `dlcnirvuyfv0msit` 已核验八卡同规格硬件；native API 三次为 18.630807 / 18.334747 / 18.463952 秒，TAEH3 为 11.488901 / 11.580645 / 12.061628 秒，中位数分别 18.463952 / 11.580645 秒。
SP8 三次 raw shape、finite、decoder parity 与重复性亦通过。SP8 warm scheduler 1440 帧耗时 89.261799 秒，首帧 1.369794 秒，16.132321 FPS，chunk p50/p95 1.022606 / 1.520293 秒，累计卡顿 29.098696 秒，持续 24 FPS 门槛失败。

### Trace、Perfetto 与资源用量

SP1/2/4 共 14 个 rank trace 已全部离线解析，`read_errors` 均为空。SP1 native 的 CUDA
kernel inclusive 总量约 50.565 秒，命名 `video_decode` 区间约 33.761 秒；SP1 TAEH3 的
`video_decode` 区间为 58.210 毫秒，CUDA kernel inclusive 总量约 16.863 秒。SP4 TAEH3
trace 中后段 causal block 随 `history_rows` 增长到约 1.3 秒，与长会话后段 chunk 变慢一致。

2.04 GB 的原始 SP1 TAEH3 trace 在 Perfetto 中可以载入，但显示
`SLICE_MAX_DEPTH_EXCEEDED=7090` 与
`SLICE_SPILL_OVERLAPPING_COMPLETE_EVENT=72101`。原始文件完整保留用于数值分析；另生成
只保留 CUDA kernel/memcpy/memset 与 Zing/STAGE annotation 的 249 MB 展示视图。展示视图保留
原事件时间、时长、名称和参数，在 Perfetto 实际打开后无导入错误标记。SP8 trace 分析作业在
完成 8 个 native rank 后超时，TAEH3 rank 未形成完整分析，不把它列为 SP8 归因证据。

本轮所有成功、失败和恢复作业累计 Running 时间约 **8.725 GPU·h**；从 EnvPreparing 到结束
估算约 **9.020 GPU·h**。这是分配 GPU 数乘作业墙钟时间的实验用量估计，不是账单金额。

同步画质、持续流式和 WebSocket 汇总报告：
<https://review-cdn.loopit.com.cn/world-model/sglang/evals/zing-h3-dmd/taeh3-main-20260921-r1/review/review-matrix-r8.html>
