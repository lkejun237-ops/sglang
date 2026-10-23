# Zing 固定输入客户端基线

客户端测量 TTFF、E2E、FPS、TPOC、ICL；服务端阶段计时见
[阶段计时说明](STAGE_TIMING.md)。参考官方
[serving benchmark](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling#benchmark)
的常驻服务、预热、重复请求及逐请求记录方式；协议差异使此入口独立于文本
benchmark，不将 token 指标或视频任务轮询直接套到 Zing。

## PR 性能测试要求

后续 PR 与测试使用这套英伦维护的工具，分别覆盖 **480、736 两档**，记录实际
宽×高与 case 身份；默认 case 集合不自动代表已经覆盖两档。不得将 704/720 当作
736，也不得静默修改冻结 case 的输出尺寸。需要另一档输入时使用显式 manifest /
case-set，保留来源和尺寸差异；入口或部署不支持时如实记录未验证及原因。

每个对 Zing H3 速度或显存有影响的 PR，在相同硬件、checkpoint、输入、采样及部署配置下
对比 baseline 与改动版本，同时提供端到端 TTFF、E2E、FPS（全程与首包后分别标注）、
TPOC、ICL 中位数/P95 与样本量，以及 **Encoder、DiT、Decoder 三部分各自的性能与显存**。
即使只修改其中一部分，也报告另外两部分，以识别成本转移。
组件数据遵循 [阶段计时说明](STAGE_TIMING.md)，注明逐 rank、Host/CUDA 边界与
采集开销；重叠阶段不相加成 E2E，microbenchmark 不能替代端到端结论。

连续 source 使用 `bench_streaming_source.py` 及 `build_streaming_report.py`，补充
source-to-output 延迟、tail drain、credit wait、播放停顿等适用指标。有限输入与
连续 source 分开比较。保存命令、源码 SHA、配置、预热/正式次数、失败与未执行数、
原始 JSON 和报告；未采到的指标标为未测。

## 当前默认采样口径

新测试统一使用 **前三个 target block 各四步 `[1000,750,500,250]`，后续每个
block 两步 `[1000,500]`**（few_step，initial_block_count=3）。这是 block 的
去噪步数，不是只在整个请求开头运行四次。`run_benchmark.py` 显式固定该部署配置，
并在 `deployment.json` 的 `sampling_schedule` 与 `server_command` 中记录。

服务的部署采样配置优先于请求字段，因此 `sampling.json` 中的
`num_inference_steps=4` / 四步列表不能证明实际全程四步。连接已有服务的
`bench_client.py` 与 `bench_streaming_source.py` 不修改服务配置：测试前核对启动参数，
连续 source 还需核对 admission 解析后的 profile/CLI 配置；显式 profile 或 CLI
可覆盖默认值。复现全程四步须显式设置 `--zing-initial-block-count 0`、
`--zing-solver few_step --zing-denoising-step-list 1000,750,500,250`。

历史全程四步、4→1 及其他采样实验保留原始参数，不能改标为 4→2 或直接混作
同条件 baseline。新测试的 baseline/candidate 必须使用相同采样配置；480 与 736
分别验证，未实测的当前版本性能和画质标为未验证。

## PR 三部分报告格式

480 和 736 的每个 case 各填一组下表，baseline / candidate 使用相同计时与显存口径。
耗时至少报告每请求累计阶段耗时的中位数/P95、请求样本量及相对变化；若报告每次
调用或每 chunk 的分布，单列其边界和样本量，不能混作请求样本。多卡逐 rank 列出，
可在摘要中取最慢 rank，但不能把 rank 当成独立请求或将其耗时相加为延迟。

| 部分 | 耗时 baseline → candidate（中位数/P95、n、变化） | 常驻显存 baseline → candidate（GiB、变化） | 请求峰值 baseline → candidate（GiB、变化） | 口径 / 未测原因及补测计划 |
| --- | --- | --- | --- | --- |
| Encoder（Qwen / reference VAE / source VAE 分项） | 待填写 | 待填写 | 待填写 | Host 或 CUDA；进程/设备/rank 与测量阶段 |
| DiT | 待填写 | 待填写 | 待填写 | 包含 KV、AdaLN、Graph 等常驻分配的实际范围 |
| Decoder（注明实际后端，如 Lynn） | 待填写 | 待填写 | 待填写 | 明确是否含输出搬运/等待 |

显存的常驻值取模型加载、配置的 KV/Graph 预热完成后、请求开始前的快照；请求峰值
明确是否在该请求开始前重置峰值统计。只能取得进程 lifetime peak 时按原名标注，
不能当作请求峰值。allocated/reserved 分列，不相加；driver 整卡采样需注明采样周期，
其采样最大值也不等于瞬时峰值。加载峰值影响能否启动时另列，不用暖机后快照替代。

组件共进程时，allocator 统计属于进程，不能直接据其峰值声称组件独占；Qwen 与
Encoder 共卡时，driver 数据属于整卡。保留这类共享范围并标明“组件独立显存未测”，
不能简单相减或把三部分的峰值相加。工具尚无独立组件显存或远程 Qwen 计时能力时，
如实写明缺项、原因和补测计划；“未改动”不能代替实测，也不据此声明三部分验收通过。

整体 E2E 使用关闭诊断采集的正式轮；阶段耗时/显存诊断可另采同配置请求，注明
采集开销及覆盖范围。复用已有报告和原始 JSON 的一个入口即可，不重复归档日志。
现有阶段回传与 rank memory snapshots 不自动证明三部分均可独立归因；边界详见
[阶段计时与显存](STAGE_TIMING.md#三部分显存与性能的归因边界)。

## 从 checkpoint 一次运行

在已安装 SGLang 的 NVIDIA 环境中运行：

```bash
python3 benchmark/zing_h3_causal/run_benchmark.py --model-path /path/to/sglang-checkpoint
```

读取 checkpoint 的 `zing_config.json` 和 `zing_realtime_profile.json`，从 profile
提取四步 schedule、solver、guidance，seed 默认为 123。默认启动本机 SP2 常驻服务，
四个 case 各预热一次、正式三次，串行执行。完成后在
`benchmark-results/<时间>/` 保存 `report.html`、`report.json`、客户端记录、配置与
原始 RGB；服务及其 worker 在结束时自动退出。

素材默认缓存于 `~/.cache/sglang/zing-benchmark/`。缺失素材从 manifest 的 S3 URI
下载，需要 `fsspec/s3fs` 和该 bucket 的读取权限；已有本地素材可用
`--media-root /path/to/media`。下载、SHA 校验和启动均在请求计时之外。

`--refresh-cache` 从 manifest 原始 URI 重新获取所选素材，同一 URI 在集合内只刷新一次。
下载到唯一临时文件，字节数和 SHA256 验证通过后原子替换；校验失败或下载中断时
本次准备失败，旧缓存保留，不回退继续运行。`--media-root` 是缓存目标目录，
刷新不会更新已有的上游 bucket mirror。
`media_preparation.json` 记录每项素材的原 URI、预期字节数/SHA256、
`hit` / `miss` / `refresh`、校验状态和准备耗时；失败时也保留回执。
成功准备的信息同时写入 deployment 配置，随结果 JSON 保存，准备耗时不计入 E2E。


有 `pipeline_config.json` 时读取 checkpoint 内的部署覆盖配置，其中相对 TAEH3
路径相对于 checkpoint 解析；未附带时使用 native decoder。可通过
`--pipeline-config` 指定现有覆盖文件。默认 decoder 因此可能不同于下方 TAEH3 实测，
实际配置写入结果文件。

常用选项：`--case 5s|30s-wasteland|30s-town|30s-indoor`、`--sp-degree 1|2`、
`--warmup-runs`、`--measured-runs`、`--output-dir`。阶段采集默认关闭，用
`--stage-timing host|cuda` 开启；`--preflight-only` 只准备素材、核验配置。
开启阶段采集时，入口自动请求服务端回传各 rank 的 profile，保存在
`client/profiles/<请求 ID>.json` 并合入同一份 `report.json`，无需再取服务端文件。

冻结源码归档没有 `.git` 时，使用 `--source-revision <完整 SHA>` 记录版本。

该入口仍采用完整 source 的离线输入。动态 source 传输与 #194 的流式指标链路一起
推进，接入后复用相同的客户端结果格式。

## 从远程服务接收阶段数据

benchmark 默认请求 Host 阶段数据；无需设置计时环境变量。普通生成请求用
`return_stage_timing=true` 同时开启采集和回传，不传则不采集。
`--receive-stage-timing` 保留为显式开启参数；`--no-stage-timing` 关闭 benchmark 采集。

服务启动时自动创建部署独立的临时目录，各 SP worker 共用该目录；独立 Encoder
通过内部 RPC 收到相同目录。`SGLANG_ZING_STAGE_TIMING_DIR` 可选指定父目录，
服务只清理自己创建的子目录。

生成与清理完成后，服务端发送 `profile_result`。客户端先保存到 `profiles/`，
再发送一次 `profile_ack`；服务端删除本请求计时文件。断连、ACK 超时、失败或
取消也做清理，服务退出时删除整个自动目录。回传与确认等待不计入 TTFF/E2E。
已有 `profile=true` 保持 Torch profiler 含义，不用于开启这份 Host 计时。
报告生成器读取客户端结果，各 rank 数据保持独立。

## 选择用例集合

默认集合为 [`whitebox-basic`](case_sets/whitebox-basic.json)，包含下面的四组输入，由 Xiaoxiao 提供。
用例保留各自 manifest 内的原始请求与来源信息。

```bash
python3 benchmark/zing_h3_causal/run_benchmark.py --model-path /path/to/sglang-checkpoint --case-set whitebox-basic
python3 benchmark/zing_h3_causal/run_benchmark.py --model-path /path/to/sglang-checkpoint --case-set /path/to/my-cases.json
```

自定义集合只需 `name` 和有序 `cases`，每项填写 `manifest` 与 `case_id`，格式见默认集合。
manifest 路径相对于 SGLang 仓库根目录解析。集合按顺序串行运行，每个 case 单独统计；
素材准备与缓存复用现有流程，无需逐 case 执行准备命令。`--case` 仍用于单 case 调试，与 `--case-set` 二选一。
客户端入口 `bench_client.py` 同样支持 `--case-set`，其素材路径仍需对服务端可读。

其他工具可调用 `case_sets.load_case_set(name_or_path, checkout=...)`，取得集合定义、
成员 manifest/ID 及原始请求。集合定义与 SHA256 写入客户端 `config.json` 并随报告 JSON 保存。

## 固定输入与部署

| case | 固定身份 | 输出 | 参考条件 |
|---|---|---|---|
| `5s` 海岸 | `zing-whitebox-video-five-triview-0001` | 120 帧、24 FPS | 六图 |
| `30s-wasteland` 废土 | `zing-whitebox-video-30s-six-triview-0001` | 720 帧、24 FPS | 全局图 + 六张三视图，共七图 |
| `30s-town` 地中海小镇 | `zing-whitebox-video-30s-six-triview-0002` | 720 帧、24 FPS | 全局图 + 六张三视图，共七图 |
| `30s-indoor` 刺绣霓虹漫画／室内通道 | `zing-whitebox-video-30s-two-triview-0001` | 720 帧、24 FPS、832×480 | 全局图 + 主角／林姐三视图，共三图 |

PR #299 的室内用例保留原始三图条件与 prompt；素材及请求准备已验证，GPU 生成效果待实测。
人工检查重点为人物身份、入镜／离镜时序、参考下排外观与长期闪烁／漂移。

2026-10-12 按用户确认采用 PR #261 的两条准确七图长视频输入。
默认 `--case all` 串行执行四个独立 case；长视频使用
`whitebox_video_long_cases.jsonl`（可用 `--manifest-long` 指定）。请求及条件必须与仓库
规范 case 一致，不能裁剪、循环、减少参考或改写 prompt。旧 `--case 30s` 的六图 ID
保留为显式旧入口，缺少其 manifest 时仍明确失败；它不进入默认矩阵，也不会被七图冒名替换。

使用完整 source 的有限视频接口 `/v1/realtime_video/generate`，不使用
`source.append`，不模拟实时输入。原始 request 的 source、prompt、各 case 参考图及其
conditions 映射保持不变，只将媒体 URI 映射至已核验的本地资产。
`--media-root` 指向 S3 bucket 的本地镜像根目录，保留 `world-model/...` 层级。
客户端逐份核验 manifest 声明的字节数和 SHA256。相同绝对媒体路径必须对
服务端可读；客户端传输的是完整请求及 URI，不是上传媒体文件内容。
素材复制/下载不计时，服务端实际读取和编码耗时计入请求延迟。

服务须已加载模型并完成启动预热。每个 case 再执行完整请求预热，随后重复
正式测量；所有请求串行，前一个请求正常关闭后才开始下一个。请求内部的
多卡/流水线设置属于部署配置，此工具不改动。失败后停止，不自动重试或降配，
避免在未确认清理状态时启动下一请求。

## 运行

依赖仅 `msgpack`、`websockets>=14`；本地测试另需 pytest，不需要 CUDA/PyTorch。
提前准备采样参数与部署 JSON，例如（**示例参数，需匹配实际 checkpoint**）：

```json
{
  "seed": 42,
  "num_inference_steps": 4,
  "guidance_scale": 1.0,
  "zing_solver": "few_step",
  "zing_denoising_step_list": [1000, 750, 500, 250],
  "zing_realtime_decode_mode": "stable"
}
```

部署 JSON 记录实际服务源码 SHA、checkpoint 身份、GPU、并行/worker 布局、精度、
offload 与编译设置；这是运行者提供的记录，不冒充客户端自动核验的服务配置。
工具固定原尺寸 RAW RGB 输出、关闭输出 pacing、保持有限接口要求的完整历史；
不强加 GPU 优化开关。不支持有声输出，音视频请求明确拒绝。

```bash
python3 benchmark/zing_h3_causal/bench_client.py \
  --url ws://127.0.0.1:30000/v1/realtime_video/generate \
  --media-root /data/leap-worldmodel-thailand \
  --sampling-params /data/benchmark/sampling.json \
  --deployment /data/benchmark/deployment.json \
  --output-dir /data/benchmark/run-001 \
  --warmup-runs 1 --measured-runs 3 --save-media
```

单 case 用 `--case 5s`、`--case 30s-wasteland`、`--case 30s-town` 或 `--case 30s-indoor`。
`--preflight-only` 核验输入并保存配置，不连接服务。输出目录必须是新目录。
源码以冻结归档部署、没有 `.git` 时，显式传 `--source-revision` 的完整 Git SHA；
修改后的客户端另以文件 SHA256 记录，部署方仍须校验归档和覆盖文件。
`--save-media` 保存原尺寸 RGB；后台写盘队列有界，跟不上会明确失败而非阻塞
接收后继续给出正常成绩。不保存时仍逐 payload 验证尺寸、格式、字节数和顺序。
保存文件可用 ffmpeg 的 rawvideo/rgb24 输入选项按请求尺寸及 24 FPS 播放或封装。

## 指标边界

客户端单调时钟；连接、素材准备及 init 序列化在计时外。`t0` 是发送 init 前。
完整二进制消息收到后立即打点，早于解包、校验和写盘；header/payload 分包时
取完整 payload 到达时刻。控制包、错误包与心跳不算媒体。

一个客户端 chunk 是一个完整 RGB `frame_batch`，不是 TCP 包，也不是 DiT block。
服务端一个 `chunk_index` 可能拆成多个 `frame_batch_index`；每批分别记录到达时间，
保留两种编号，避免把合法拆包当重复。有限接口没有逐帧帧号时，依据连续的
block/batch 编号重建帧范围，并明确标记 `frame_start_origin`。

| 指标 | 定义 |
|---|---|
| TTFF | 首个有效 batch 到达时间 − t0 |
| E2E | 最后有效 batch 到达时间 − t0 |
| `fps`（原始全程口径） | 实际有效帧数 / E2E，包含首次等待 |
| `post_ttff_fps`（网页 FPS） | (总帧数 − 首批帧数) / (E2E − TTFF)，不足有效时间区间为 null |
| ICL | 相邻 batch 到达间隔；记录原始值、样本量、中位数/P95 |
| TPOC | 首 batch 后平均 ICL；不足两个 batch 为 null |
| IFL | null；没有独立逐帧到达时间，不插值伪造 |

RAW RGB 完整 payload 经检查即为可解码像素，不包含浏览器呈现时间。
有限接口以 close code 1000、reason `generation complete` 表示完成，发生于清理后。
必须同时满足完整帧数、完整 split batch 和正常完成关闭；其他正常关闭原因也失败。
`completion_seconds` 单独记录，绝不加入媒体 E2E。

## 结果与验收

### 离线网页报告

每次测量后，从本次输出目录生成一个可直接打开的 HTML；没有 GPU、模型或前端依赖，
也不需要启动网页服务。报告重新从逐 chunk 到达时间复算指标，不引用旧实验数值。
同时自动保存同名结果 JSON，例如 `report.html` 与 `report.json`。

```bash
python3 benchmark/zing_h3_causal/build_benchmark_report.py \
  --run current=/data/benchmark/run-001 \
  --output /data/benchmark/run-001/report.html
```

多个配置可重复传 `--run name=/path/to/client`，在同一页分别比较；每份目录包含
`config.json`、`requests.jsonl`、`chunks.jsonl`，预检或未执行请求显示未执行数量。
报告保留所有预热、失败和部分输出，但正式指标只汇总正式成功请求。
布局、配色、图表、原始 chunk 表及 CSV 下载沿用已有 `view/5s-results.html`。
网页顶部可“导出当前结果 JSON”，或一次打开多个结果 JSON；使用“结果文件”
选择器切换查看，不把不同人的结果合并统计。不同文件可以包含不同 case。
读入时核对格式版本、请求身份、实际帧数和到达顺序，并复算客户端指标及阶段累计值。
错误文件显示错误，其他正确文件仍可打开；未读入的结果不覆盖当前内容。
网页和 JSON 都可以本地保存或直接发给其他人，无服务器、上传或多端同步。
命令行可用 `--json-output /path/to/results.json` 指定 JSON 的保存位置。
同一结果文件可包含多个 case，使用原有“结果文件”选择器逐 case 切换；每个视图只比较同一个 case 的不同采集模式，不混合统计。
网页 FPS 沿用该页的首包后口径：首包之后新增帧数 / (E2E - TTFF)，先逐请求计算，
再对正式成功请求取中位数。原始 benchmark 的全程 FPS（总帧数 / E2E）不改写，
在网页原始计算详情中保留两者的区别。
未运行或失败的 30s 不会自动变成完成；显示真实请求、失败与未执行数量。

可选阶段记录默认读取 client 目录旁的 `timing/`；其他布局显式指定
`--timing current=/path/to/timing`。同一主机时钟域的双 rank 沿用原页面的共用 host
时间轴；不同主机/时钟域不绘制共用时间轴，仅按货号关联，不跨机器相减。
显示逐阶段 host/current-stream 累计值，缺失值为未测；Conditioning 包含 Encoder，
不计算相加 E2E 或阶段百分比。health 损坏或丢样明确展示；此页面不是严格 ledger
验收器，不替代 `analyze_stage_timing.py`、完整媒体解码或人工观看。

HTML 内含本次输入、素材路径、部署配置和原始测量数据，按与原始工件相同的
内部访问范围保存；本工具不上传或发布网页。

- `config.json`：输入与资产身份、完整 init、采样/部署和客户端版本、计时口径。
- `requests.jsonl`：含预热、正式成功及失败，部分输出指标也保留但不进入成功汇总。
- `chunks.jsonl`：逐请求/batch 到达时间、帧范围/数量、媒体时长及 ICL，可独立复算。
- `summary.json`：各 case 分别统计，不混合；预热排除，失败及未执行数保留。
- 可选 `.rgb`：逐请求原始媒体，包含失败请求已收到的部分。

主线验收：固定输入正确、串行完整运行、指标可复算、失败不消失。
没有 24 FPS、固定提升百分比或零误差门槛。少量运行的 P95 只是描述性统计，
始终带样本量，不宣称稳定尾延迟或生产实时能力。

持久测试只保留公式、计数、边界和固定输入的短合同测试：

```bash
python3 -m pytest -q benchmark/zing_h3_causal/tests/test_bench_client.py
```

开发时临时验证 header/payload 延迟、控制包、断流、超时、慢写盘；不维护新的
模拟服务框架。真实 GPU 各 case 的验收另跑，CPU/模拟流通过不能替代它。
各 case 的完整业务验收必须依据各自真实运行与原始记录，CPU 合同通过不代表 GPU 已通过。

本阶段的实际运行、验收状态和工件身份见
[阶段性验收记录](CLIENT_BENCHMARK_VALIDATION.md)。


## 连续 source 的客户端起点与失败回执

`bench_streaming_source.py` / `continuous_source_client.py` 使用真实 source 流传输。
其 `timing_origin=client_start_before_connect_and_source_decode` 包括建连和 source
解码启动等待，与上述有限请求 init-send 起点分开记录。`timing_boundaries` 明确
每个时间点；`init_send_to_rgb` 另外保留 init-send → 解包前完整 RGB 的兼容指标。
双方都按首个和最后一个有效 batch 计算 TTFF/E2E，不把 EOS、profile 回传或收尾
时间算进媒体 E2E。`elapsed_seconds`、`completion_overhead_seconds` 单独记录收尾。

接收记录在 RGB 校验后、写盘前保存，`sink_confirmed_bytes` 另计写盘成功字节。
收到全部 RGB 后若 EOS 丢失、sink 写入失败或收据损坏，请求仍失败并排除正式汇总，
但保留 `client_observation`、完整/部分交付记录和原始异常，便于区分生成与收尾问题。
旧结果没有新字段时显示缺测，原始 `fps` 的语义保持兼容。

全50层三目标的场景、运行版本、phase 与算子口径，以及离线报告重建命令见
[完整链路实测](STAGE_TIMING.md#全50层完整链路实测2026-10-19)。
