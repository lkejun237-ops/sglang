# PR275 / PR283：客户端与阶段 benchmark 历史交付

本文保留当时的部署、指标和验收结果。当前综合工具入口见 [README](README.md)，
现行计时与报告使用见 [STAGE_TIMING](STAGE_TIMING.md)。历史 TAEH3/CUDA 模式不代表当前部署默认值。

## PR #275

对应 issue #253，主要分客户端和服务端的 benchmark。

- 客户端：TTFF、E2E、FPS、TPOC、ICL。
- 服务端：Conditioning、Encoder、DiT、Decoder 的 host / CUDA 时间。

约束条件：模型常驻、完整请求预热后测量、请求间串行、并发 1。已有实测采用 **SP2**。

> issue #253 要求 source Encoder 与 DiT 卡组分离。已有数据来自共享卡组：每张
> SP2 卡上同时执行 Encoder、DiT 和 Decoder。拆分部署由对应 PR 推进。

### 客户端

实现：新增 `bench_client.py`，发送固定业务 payload，记录发送时间和收到每组
frame batch 的时间戳。#275 完成 5s 基线，#283 补齐两个 UniEdit 30s case，并纳入 #299 的室内 30s case。

设 `t0` 为发送 init 前的时刻，`t1` / `tn` 为首包 / 最后有效媒体包的到达时刻，
`N` 为实际输出帧数，`K` 为媒体包数：

| 指标 | 公式 |
|---|---|
| TTFF | `t1 - t0` |
| E2E | `tn - t0` |
| 全程 FPS | `N / E2E` |
| 首包后 FPS（#283 网页展示） | `(N - 首包帧数) / (tn - t1)` |
| ICL | 相邻有效媒体包的到达间隔 |
| TPOC | `(tn - t1) / (K - 1)`，即平均 ICL |
| IFL | 当前一包可含多帧，没有逐帧时间戳，记为未测 |

E2E 截止最后媒体包，会话关闭与清理另记。本次客户端和服务端同节点，通过
loopback WebSocket 通信；测量包含该链路交付时间。

### 服务端

在 CPU host 调用前后打点，并使用 CUDA event 记录当前 stream 的时间区间。

- Conditioning：`codecs.prepare()` 中的媒体读取、预处理、条件准备与 VAE 编码。
- Encoder：Conditioning 内部的 Qwen / reference VAE / source VAE 子区间。
  当前 conditioner 回调路径没有独立 Qwen span，Qwen 包含在 Conditioning host 内。
- DiT：`_sample_realtime_block()` 整个调用。
- Decoder：`_decode_realtime_output()` 整个调用。

host 记录调用的进入、返回和交接；CUDA event 记录流上的执行区间。逐 rank 展示，
重叠区间不相加成 E2E。

## PR #283

对 #275 做四项完善：补齐用例集合、简化 checkpoint 入口、增加结果网页、回传服务端阶段数据。
其中两个 30s 输入复用 #261 的用例，本 PR 将它们纳入默认 benchmark dataset。
动态 source 传输与 #194 的流式指标链路一起推进，本次实测采用离线输入。

素材支持跨运行复用，也可用 `--refresh-cache` 从原 URI 重新获取。刷新校验通过后
原子替换，失败保留旧缓存并停止本次准备。`media_preparation.json` 及结果配置记录
逐素材的 hit/miss/refresh 与准备耗时，准备时间不计入客户端 E2E。

### checkpoint 简化入口

```bash
python3 benchmark/zing_h3_causal/run_benchmark.py --model-path /path/to/sglang-checkpoint
```

从 checkpoint 读取模型与采样配置，准备素材，启动本机 SP2 服务，执行预热和正式
请求，最后生成 `report.html` / `report.json`，不需要手写 sampling/deployment JSON。
默认使用 `whitebox-basic` 集合，每 case 预热一次、正式三次。用 `--case-set /path/to/cases.json` 可选择其他集合，只需列出已有 manifest 路径和 case ID；`--case` 保留单用例调试。可指定素材缓存目录、SP 卡数和输出目录。

素材首次下载需要对应 bucket 的读取权限，之后复用缓存。checkpoint 附带
`pipeline_config.json` 时使用其中的 decoder 配置，否则默认 native decoder；
当前三组实测采用 TAEH3，运行配置会一并保存。

一键入口的客户端与模型服务运行在同一台机器，输入素材使用双方可读的文件路径。
结果 `report.html` / `report.json` 可以拿到本机打开；独立客户端连接远程服务时，
输入路径仍须对服务端可读。

### 阶段数据回传

开启 `--stage-timing host|cuda` 后，服务端复用 session/generation/dispatch 标记，
等待各 rank 的阶段记录写完，通过原 WebSocket 回传。客户端保存到 `client/profiles/`
并合入结果 JSON，用户无需登录服务器取 profile 文件。独立客户端使用
`--receive-stage-timing`，服务端需已开启采集。profile 等待另记，E2E 仍截止最后媒体包。

### 网页可视化

沿用原 5s 网页布局，展示客户端指标、chunk 到达曲线与服务端阶段时间线。
支持导出 JSON、打开多个结果文件、切换查看和下载 CSV。数据不绑定测试人的目录。

当前实测页：
`exports/issue253-three-cases-20261012-r1/view/benchmark-report.html`（运行工件目录）。
数据与验证说明见 [三 case 验收记录](THREE_CASE_BENCHMARK_VALIDATION.md)。

### 测试用例

| case | 内容 | 输入清单 |
|---|---|---|
| 5s-1 | 海岸场景，120 帧，六图 | `benchmark/zing_h3_causal/whitebox_video_smoke_cases.jsonl` |
| 30s-1 | 废土场景，720 帧，七图 | `benchmark/zing_h3_causal/whitebox_video_long_cases.jsonl` 第一条 |
| 30s-2 | 地中海小镇，720 帧，七图 | 同上第二条 |
| 30s-3 | 刺绣霓虹漫画／室内通道，720 帧，三图 | 同上第三条（PR #299） |

室内 case 重点看主角身份、林姐入镜／离镜、参考下排外观及长期稳定性。
它已纳入默认集合；以下阿里云实测数据仍仅覆盖原有三组。

### 实测数据

阿里云 L20D、SP2 共享卡组、TAEH3 decoder，模型常驻、并发 1、全部串行。
下表为每组 3 次正式请求的中位数；阶段列是 rank 0 的累计 CUDA stream 区间。

| case | TTFF / s | E2E / s | 首包后 FPS | Encoder / s | DiT / s | Decoder / s |
|---|---:|---:|---:|---:|---:|---:|
| 5s 海岸 | 10.9239 | 20.5825 | 12.3206 | 7.0658 | 11.0590 | 0.0388 |
| 30s 废土 | 36.2012 | 100.0196 | 11.2663 | 15.9781 | 65.0952 | 0.2206 |
| 30s 小镇 | 36.8128 | 101.3725 | 11.1254 | 15.9669 | 65.3616 | 0.2244 |

9 次正式生成全部完成，视频规格与完整解码通过。Qwen 没有独立 span，阶段分析
明确标为未测；旧 Qwen 门禁失败及修正后的复核记录保留在运行目录。详细数据在网页
与 [验收记录](THREE_CASE_BENCHMARK_VALIDATION.md) 中查看。
新 checkpoint 入口也已通过阿里云实测：GPU 作业 `dlcpofbijm0f7fe4`，同一个常驻
SP2 服务完成全部 12 次请求，自动生成 HTML/JSON；9 个正式视频完整解码通过。
本轮阶段采集关闭，使用已有素材缓存与 checkpoint 附带的 TAEH3 配置，正式中位数如下：

| case | TTFF / s | E2E / s | 首包后 FPS |
|---|---:|---:|---:|
| 5s 海岸 | 10.8470 | 20.0117 | 13.2327 |
| 30s 废土 | 35.3813 | 98.0238 | 11.4778 |
| 30s 小镇 | 36.4663 | 99.4789 | 11.5101 |

简化入口本轮结果网页：
`exports/issue283-checkpoint-entry-20261012-r1/view/report.html`（运行工件目录）。
