# 白膜逐帧耗时分解：开发者使用说明

该工具在客户端进程内运行 Room，向已部署的 Encoder / DiT / Decoder 发送录制白膜和玩家输入，生成 JSON、HTML 和 PNG。客户端不需要 GPU，也不启动真实 Whitebox CLI；真实推理模式使用远端 GPU。它会占用一个活跃会话，应在实例空闲、已获得使用授权时运行。结束后释放本次会话，不停止或重启已有 Worker。

## Codex 一体化入口

项目提供 [`sglang-latency-breakdown`](../../.agents/skills/sglang-latency-breakdown/SKILL.md)，
把实例实测、数据核验、报告生成串联起来；已有 JSON 时可只重绘，不重新占用 GPU。

- 实测示例：`使用 $sglang-latency-breakdown，测试指定的 480 实例并生成延迟分解报告。`
- 重绘示例：`使用 $sglang-latency-breakdown，只重绘这份 latency-breakdown-data.json，保留数据来源。`
- 要求分享链接时，另复用已有报告发布流程；mock 演示和真实模型实测必须明确区分。

采集和可视化仍是两个独立程序：`run_latency_breakdown_replay.py` 默认在采集后生成报告，
`latency_breakdown_visualizer.py` 可独立读取 JSON 重新生成 HTML/PNG。skill 负责选择与串联，
不增加另一套数据采集实现。目前范围是 Zing H3 三角色 source 流式路径。

## 环境和版本

- 使用包含 PR #690 及计时修复的 SGLang checkout；客户端需要项目的实时服务依赖、Pillow，以及 FFmpeg / ffprobe（视频编码需 libx264）。运行命令均在仓库根目录执行。
- 三角色服务也要运行计时修复版本。仅更新客户端，无法补回旧服务未记录的传输等待。保存各角色完整源码 SHA、镜像、checkpoint、GPU、精度、SP、采样与并行配置；`--server-revision` 只记录部署方提供的版本，工具不会自动核实它。
- `--mode mock` 是本地模拟，只验证回放和报告，不能代表真实模型性能；`--mode real-sglang` 才向部署实例发真实请求。
- 原始案例有 prompt、参考图和按键数据。报告或素材发布遵守原有分享授权，不将凭据写入命令或产物。

## 准备 480 / 736 素材

输入为 WTP 本地归档根目录，需包含 `replay-input/plan.json`、参考图，以及 `archive/<game>/streams/<stream>/whitebox.mp4` 和 `inputs.jsonl`。脚本目前选择排序后的第一个 game / stream；准备目录时只保留目标案例。

```sh
export PYTHONPATH=python
export SGLANG_LIGHTWEIGHT_RUNTIME=1

python benchmark/whitebox/run_latency_breakdown_replay.py \
  --extract-from-wtp "$WTP_ARCHIVE" --extract-only \
  --clip-start-s 3 --clip-duration-s 16 --capture-fps 30 \
  --width 832 --height 480 --resolution 480 --clip-dir "$CLIP480"

python benchmark/whitebox/run_latency_breakdown_replay.py \
  --extract-from-wtp "$WTP_ARCHIVE" --extract-only \
  --clip-start-s 3 --clip-duration-s 16 --capture-fps 30 \
  --width 1280 --height 736 --resolution 736 --clip-dir "$CLIP736"
```

两个目录分别保存 `case_meta.json`、`whitebox.mp4`、`inputs.jsonl` 和参考图。`--width/--height` 只控制截取；已有素材的回放尺寸来自 `case_meta.json`，`--resolution` 用于校验，不会偷偷缩放。480/736 是像素高度，报告同时记录请求的完整宽×高。每档使用独立输出目录，不合并统计。

1280×736 的重新缩放案例属于自定义诊断输入，**不是**已经冻结的 `product-case-20261029` 736 集合。正式性能验收继续使用 [统一 benchmark](../zing_h3_causal/CLIENT_BENCHMARK.md)；当前产品集合仅冻结 832×480，736 需另行冻结输入及更新边界。

## 默认：复用集成测试的三地址直连

复用 Room–SGLang 集成测试的 `source_replay_role_pool.SuiteRolePool`，在客户端进程内建组，
不需要部署 Coordinator、Room HTTP 或 Redis；Room 供帧仍使用真实 `SourceSession`。
仅适用于未注册到另一 Coordinator 的隔离测试 Worker。部署方需在客户端安全配置 `ROLE_POOL_SIGNING_SECRET`，与三个 Worker 的 control epoch 和签名配置一致；工具不会向服务索取密钥。

```sh
python benchmark/whitebox/run_latency_breakdown_replay.py \
  --mode real-sglang --clip-dir "$CLIP480" --output-dir "$OUTPUT480" \
  --resolution 480 --num-inference-steps 2 \
  --encoder-url "$ENCODER_HTTP" --dit-url "$DIT_HTTP" --decoder-url "$DECODER_HTTP" \
  --profile 480 --server-revision "$DEPLOYED_SGLANG_SHA"
```

这些是管理 HTTP 地址，不是输入/输出 WS。三角色须声明一致的 compatibility、network domain、profile limits；所选 profile 必须同时支持案例实际宽×高。736 同样显式使用对应素材与 profile。不把该直连模式用于接管线上已受 Coordinator 管理的实例。Coordinator 模式与直连模式互斥。

## 可选：测试已挂载 Coordinator 的实例

由部署方提供 Coordinator 管理 HTTP 地址、该模型的路由 revision、profile、compatibility、network domain 和已配置的 Room token。token 通过既有安全方式设置到 `WHITEBOX_ROOM_TOKEN` 环境变量；不要把它写进下面的命令、报告或 Git。

```sh
export WHITEBOX_ROUTING_REVISION="$MODEL_ROUTING_REVISION"
python benchmark/whitebox/run_latency_breakdown_replay.py \
  --mode real-sglang --clip-dir "$CLIP480" --output-dir "$OUTPUT480" \
  --resolution 480 --num-inference-steps 2 \
  --coordinator-url "$COORDINATOR_HTTP" --profile 480 \
  --compatibility "$MODEL_COMPATIBILITY" --network-domain "$NETWORK_DOMAIN" \
  --server-revision "$DEPLOYED_SGLANG_SHA"
```

736 使用 `CLIP736`、独立的 `OUTPUT736`、`--resolution 736 --profile 736` 以及部署方提供的 736 路由 revision。profile 名以实际部署为准，不能仅修改标签就让 480 实例支持 736。工具先校验路由；缺少相应 profile、无空闲会话或鉴权失败时应保留失败信息，不修改共享路由或回退成 480。

客户端必须能访问 Coordinator 和服务返回的输入/输出 WebSocket 地址。管理接口能访问不代表媒体路径已通。不要把产品页面 URL 或 Decoder 输出 WS 当作 `--coordinator-url`。

## 按会话开启计时与同版本对照

普通模型请求的 `return_stage_timing` 默认 `false`。在 WebSocket `init` 设置 `true` 后，
Encoder 将选择传给 DiT 和 Decoder，仅该会话采集详细逐帧计时。下一会话重新选择，
不支持在推理中途翻转开关；部署更新后不需要为了每次开/关重启 Worker。

本诊断工具默认 `--stage-timing on`，自动把字段加到 Room 的模型 init；
同一实例、素材和参数改为 `--stage-timing off` 即是不采集详细计时的同版本对照。
关闭时不记录新增采样/去噪 CUDA events、不构造逐帧计时或发送 `chunk_timing` / `chunk_telemetry`；
既有传输同步、业务消息及 Room 自身观测保留。只对独立测试会话做对照，不修改活跃产品实例。

CUDA event 的 `record` 是异步入流，但仍有记录和消息处理开销。读取耗时复用传输完成边界；
Encoder 将已有发送前 stream 同步前移并用标记避免重复。具体成本须以同条件 GPU A/B 为准，
不能从“没有新增入口同步”推导为“零开销”。该字段也启用既有 stage timing，A/B 代表整个开关的成本。

## 隔离 GPU A/B 复现

`latency_breakdown_gpu_ab.py` 复用原成功 B300 作业的 E1+DiT SP2+D1 启动配置，
在同一组内按分辨率执行 off/on 预热，再交错执行 `off,on,on,off,off,on`（每臂三次）。
两臂复用相同源码、模型、输入、seed、2 step、精度、编译、缓存和自然供帧策略；
只改变 `--stage-timing`。`736` 是 480 白膜 bilinear 派生的诊断素材，独立统计。

执行输入为已推送源码的 `source_bundle.tar.gz`（含可校验的 Git 身份）、
`wtp_clip_16s.tar.gz`、`run-plan.json`，以及只读挂载的已验证 prepared runtime 和模型。
plan 指定 `source_revision`、`image`、`model_path`、`lynn_path`；不包含凭据。
在匹配镜像中先以零 GPU 运行 `latency_breakdown_dlc.sh --prepare-only`，
核验导入来源、默认开关、素材和 FFmpeg，再在已授权四卡隔离任务运行同一脚本。
这不是发布或创建 Coordinator 的入口，也不连接产品实例。

保留每臂原始 JSON/HTML/PNG、运行配置、日志、对比 JSON 和 driver 显存的 1s 采样。
`room_output_elapsed_s/fps` 从 render.start 计到最后一次 Room model media callback，
排除 prefill 和离线报告渲染；它不是浏览器显示 FPS。供帧是自适应的，实际选帧可能不同，
必须连同输出帧数及 source-to-output 延迟一起判断。显存采样是整卡 driver 口径、峰值下界，
不是 PyTorch allocated/reserved，也不能拆成 Qwen/参考图 VAE/source VAE 独占显存。
关闭臂没有详细阶段计时，其分项记为未测，不能填 0。

收尾结果与测量结果分别验收。当前 `a1dd8ec9c2` 的隔离实验完成 12/12 正式请求，
但停止进程后的缓存链接/挂载保护拒绝清理，作业为 Failed；不绕过保护、不宣称整个套件通过。
后续脚本将测量完成与清理状态分开写入，并在保护拒绝时保存诊断信息。

## 如何读结果

- `latency-breakdown-data.json`：schema v2，保存逐帧数据、各指标中位/P95/均值、有效 `count`、`missing_count` 和 `coverage`。缺失值为 `null`，不作为 0 纳入统计；真实的 0 仍是有效样本。
- `index.html`：可缩放、选帧的报告。输出点保留 Room 实测到达时刻。三角色条仅是**串行耗时示意**，不是实际阶段起止时间，不能据此推断流水线重叠或收益；缺少完整阶段数据时不画推算条。
- `latency-breakdown-overview.png`：便于对照的静态概览；未测项显示“未测”。
- summary 记录请求尺寸、resolution、供帧/模型 FPS、案例元数据与按键文件 SHA256、客户端 Git SHA/工作区修改状态，以及部署方提供的 server SHA。完整运行条件仍需与部署回执一起保留。

Encoder `return_ms` 从生产者提交发送起计至 DiT 接收完成，包含发送排队、传输和接收背压。只在两端可确认共享 monotonic clock domain 时计算；跨节点、隔离 boot namespace 或旧服务缺少时间戳时标为未测，不用未校准的墙钟相减。macOS 无 boot ID 时只支持同进程通信模拟。

`process_ms` 的边界随角色而异：Encoder 是本轮 Host+CUDA 完成边界，DiT 主体使用 CUDA event，Decoder 包含解码及发布到共享内存。chunk/group 耗时会归给相应帧，不是每一帧独立发起的 GPU kernel 时间。阶段和与逐帧 E2E 不可直接相减推算网络开销。

## 指标如何采集

业务链路是 `source.append → Encoder moments → DiT latent → Decoder RGB/SHM → output WS → Room`。
`received_mono_ns` 在 Encoder API 接收 source.append 时记录，随入队项和分片保留。
它是内部单调时钟时间戳，不是新增的业务延迟指标，也不是视频 PTS。原有 `received_ns`
使用墙钟便于日志关联；耗时相减使用 `perf_counter_ns`，避免系统时间调整影响结果。

| 角色与指标 | 当前实现的测量边界 |
| --- | --- |
| `encoder.wait_ms` | 当前 `_forward_source` 开始减去原始包的 `received_mono_ns`，包括等待凑组和排队；同包多帧共用到达时刻。 |
| `encoder.process_ms` | 本轮开始至 `encoder.pop()` 的生产者 CUDA stream 完成，减去本轮之前 `send_dit` 调用的耗时；包括 Host 准备、拷贝与实际编码，发生 conditioning 时也包含它。不是纯 VAE kernel 时间；一轮多个 pop 时是截至当前 pop 的累计区间。 |
| `encoder.return_ms` | 提交发送至 DiT I/O 线程完成接收，包含 outbox、传输及接收背压；要求两端 clock domain 一致。 |
| `dit.wait_ms` | DiT 开始执行减去该 latent 的接收时刻，再减去后验采样的 Host 调用耗时；包括攒 AR chunk、缓冲和任务等待。 |
| `dit.process_ms` | 去噪前后 CUDA events 的 elapsed_time，加上该输入后验采样前后的 CUDA events 耗时；CPU 采样回退到 Host 调用耗时。采样 event 在已有下游同步边界之后读取，不在输入路径增加同步。wait 仍按 Host 调用区间扣除采样，两项不是可相加的独占墙钟区间。 |
| `dit.return_ms` | 当前 CUDA 路径取“DiT Host 开始至 latent 发送完成”的总区间减去 CUDA events 耗时；这是余量，可能含 Host、GPU 排队与传输，不能称为纯网络耗时。无 CUDA event 时退回 Host 区间划分。 |
| `decoder.wait_ms` | `submit` 入解码任务至解码线程通过 step gate 并开始处理的时间。它不包含 submit 前的 latent 接收与共享内存预留等待。 |
| `decoder.process_ms` | 解码线程开始至本任务全部 RGB 发布完成，包括 decode、RGB 转换、D2H 和共享内存发布；采用 Host 完成边界。 |
| `decoder.return_ms` | 任务解码完成至输出 WebSocket 发送结束，包括期间的消费/排队与发送；完成时间缺失时当前实现回退为 payload build + msgpack + WS 发送计时。不是浏览器收到或展示的时间。 |

`transport_encode_ms` 是 payload build 与 msgpack 序列化计时之和，`transport_write_ms`
是 WS send 调用耗时；这两项属于返回路径的诊断信息，不再加到 `decoder.return_ms` 上。

Encoder 的像素索引与 DiT latent/chunk 索引不同，`SourceChunkBridge` 根据像素覆盖范围携带
`encoder_frames`、`arrived_ns`、`sample_ms` 及仅限当前 rank 的 `sample_events`。最终每行 `frame_stage_breakdown` 用
`output_frame` 对齐输出帧；同组/同 chunk 的计算耗时会被其多个帧共享，不得把这些帧的
`process_ms` 再相加当成独立 GPU 工作量。DecoderOutputBroker 合并三角色，发
`chunk_telemetry`；Room 数值白名单清洗后发 `whitebox.model_timing`，并有界保存到 manifest。

Room 已有指标提供另一组实际观测边界：`source_to_selection_ms` 为采集至选帧，
`selection_to_dispatch_ms` 为选帧至发出，`dispatch_to_raw_output_ms` 为发出至收到原始输出，
`source_to_raw_output_ms` 为采集至收到原始输出。它们不包含后续浏览器解码/显示。
`input_capacity_blocked_ms` 的不同阻塞条件可能重叠，不能相加。本工具 `session_ttff_ms`
从 `render.start` 计至首个可解码输出，不包括此前已经完成的 prefill。

## Decoder 操作的含义

| op | 行为 |
| --- | --- |
| `append` | 从 NCCL 接收 DiT latent，验证连续性，加入当前 Decoder session 的 `_pending`；不是追加到 RGB 输出。`auto_output` 模式随即执行与 `output` 相同的提交逻辑。 |
| `output` | 将待解码 latents 移入解码线程的 job 队列，预留本地 SHM 输出并排入输出 mailbox；本身不等解码结束。有 direct output 时 RGB 由 Decoder WS 直接交给调用方/Room，不经 DiT；旧路径只向 DiT 调用方返回 SHM 引用，仍不传 RGB 像素。 |
| `chunk_timing` | rank 0 校验 generation 后，把 Encoder/DiT 计时作为 sideband 排入同一个输出 mailbox，不触发解码。`append` 描述符必须先发送，Decoder 才能接收 latent；发送完成后的 timing 当时尚不可得，因此单独发送这个 op。它不等待计时后才发 RGB，但仍受已有 mailbox 的背压约束。 |
| `source_progress` | 把源输入的 ownership watermark 和包边界映射为 `source.consumed` 信用回执，让上游释放输入/继续供帧。按完整包边界确认；不是输出帧数，也不是额外生成任务。 |
| `finish` | 声明输入 EOS 与最终帧数，auto_output 下排空解码，核对输入信用回执已排空，发送 `session.completed` 并等待发送结束；会话资源尚未由 close 释放。 |
| `close` | 关闭当前会话输出 mailbox、drain 解码、dispose stream 并移除会话/bridge 引用。它是会话资源清理，不是关闭整个 Decoder 服务；若尚有未发输出，mailbox 关闭会取消它们。完整正常结束顺序是先 finish 再 close。 |

`_collect_encoder_frame_timings` 集中管理 Encoder 同步和逐帧归因；返回的 `_stream_synced`
标记让 transport 跳过重复的生产者 stream 同步。`_queue_chunk_timing` 集中管理 timing
sideband 的 generation 校验和入队；主分发循环只保留 op 与 rank 分支。

工具当前只截取并绘制窗口前的 held-state；实际回放仍只发送窗口内事件。若窗口开始时已有按住的键，应选择已释放的起点，或先准备包含初始 held snapshot 的输入包；该限制属于 PR review 第四项，未包含在本次前三项计时修复中。

## 实测证据与当前边界

2026-11-12 核对 PR #690 作者作业 `dlc1dszcwbnx2o97`：PAI `ap-southeast-7` / workspace `384`，4× NVIDIA L20D（B300），E1+DiT SP2+D1；832×480、16s 白膜、30 FPS、2 step。日志及原始 JSON 均记录 480 源帧、478 输出帧、57 timing chunks、39 输入包，TTFF 1194.627 ms，source-to-output 中位/P95 为 508.898/665.766 ms（以原始 JSON 为准）。

证据入口：`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/latency-breakdown-20261112/`，包含 `results/HARDWARE.json`、`results/latency-breakdown-data.json`、Worker 日志、源码包和输入包；18 个 PR 改动文件与该源码包逐字节一致。旧 schema v1 报告需使用修复后的 runner 重新采集；仅重新渲染旧 JSON 不会恢复正确时间线。旧指标来自修复前埋点，不能补算遗漏的 Encoder 传输等待；不能作为本次修复后的 GPU 验证，也不是埋点开销的同条件 A/B。736、当前版本三角色完整性能/显存矩阵和埋点开销均未验证。


## 分组与缩放演示（2026-11-12 更新）

模拟模型沿当前正常配置使用 Encoder 每段 `[1,8,8]` pixel frames → `[1,2,2]`
latents，以及 DiT `[3,2]` latents（`split_first_latent=false`，每个 17px 段对应 `[9,8]` px）。
Lynn 输入始终是 latent，不是 pixel：每次 decode_step 消费 1 latent，每 5 步产出
`[1,4,4,4,4]` RGB 帧；EOS 最后一片可截短。

真实服务 `SGLANG_ZING_REMOTE_DECODE_PROGRESSIVE=1` 时逐 latent 发布 SHM，输出适配器
逐片发送，常规也是 `[1,4,4,4,4]`；默认 `0` 时可以整 AR chunk 返回，不能从 Lynn
内部 decode_step 推断 WS 包大小。新的 Room provenance 保存模型 WS 的
`output_batch_start/output_batch_frames`；报告 `output_batches` 单独画出这些实收批次。
Room 后续向播放器发的 preview/frame_batch 可以再次拆分，与该行不是同一层。
旧数据没有边界时显示“未采集”，不从时间接近或协议规律猜包。

模拟页现按实际协议攒组，Decoder 逐 latent 解码并发送 1/4 帧包，不再将整个 AR chunk
一次返回。阶段泳道每个协议组画一条耗时归因示意，标签区分 px（pixel frame）与 L（latent）；
它们仍不是真实 GPU 起止时间。黄色一直表示等输入/攒 chunk/队列等待，紫色才是 DiT 处理。

HTML 支持 0.25s/1s 快捷窗口、连续时间缩放、鼠标滚轮以指针为中心缩放、时间平移、
图高缩放和全屏；这些操作只改变显示窗口，不修改采集时间和统计结果。
