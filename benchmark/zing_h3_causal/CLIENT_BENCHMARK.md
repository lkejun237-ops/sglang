# Zing 客户端 benchmark

客户端测量 TTFF、E2E、FPS、TPOC、ICL，以及连续 source 的 Prompt-switch TTFF；
服务端阶段计时见
[阶段计时说明](STAGE_TIMING.md)。参考官方
[serving benchmark](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling#benchmark)
的常驻服务、预热、重复请求及逐请求记录方式；协议差异使此入口独立于文本
benchmark，不将 token 指标或视频任务轮询直接套到 Zing。

## 默认测试模型（2026-10-30 起）

后续测试固定使用 **1027 的 step 010000 AR + DMD diff** 模型，配置为
`num_frame_per_block=[3,2]`、`split_first_latent=false`。原始 checkpoint 的 SHA256 为
`89c3a67d77214e57d810e6a4e5554c7c919ae929db0174e6439dfadb53a1d731`。
模型位置、修正配置版本与部署前置条件见
[部署 recipe：当前测试模型](../../docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx#当前测试模型1027-step-010000)。
客户端连接已有服务时不修改模型配置，运行前须核对服务实际加载的模型身份和 split 开关，
并记录到部署回执。旧 checkpoint 或 `split_first_latent=true` 的结果保留原身份，不能改标为新模型结果。

## 默认测试集合（2026-10-30 起）

后续 Zing / SGLang 测试默认使用 [`product-case-20261029`](PRODUCT_CASE_REPLAY.md)，覆盖单图、
长 prompt、动态增减参考图和完整长会话。先按该说明准备真实素材，再连接已部署的
连续 source 服务；不传 `--case` / `--case-set` 即串行运行四条真实请求：

```bash
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
python3 benchmark/zing_h3_causal/bench_streaming_source.py \
  --url ws://127.0.0.1:30000/v1/realtime_video/generate \
  --media-root /data/zing-product-case-media --output-dir /data/product-case-run \
  --seed 42 --paced --timeout 1800 --warmup-runs 1 --measured-runs 3
```

原四个 case 保留为显式 `--case-set whitebox-basic` 的旧集合回归。
此外，针对细分场景（Prompt 切换、Block 0 L1 折返复用、多角色/三视图一致性、大场景、第一人称、第三人称与 `832×480 → 1280×736` 跨分辨率输出）新增显式集合 [`product-case-20261103`](PRODUCT_CASE_REPLAY.md#场景细分测试集合product-case-202611038-个核心-case)（含 8 个核心 case，高频切换截取前 4～6 次；预览见[在线报告](https://review-cdn.loopit.com.cn/reports/dev-api-cases-20261103/r1/index.html)），可通过 `--case-set product-case-20261103` 准备与运行。
有限输入工具 `run_benchmark.py` / `bench_client.py` 不支持新集合的动态更新，须显式
选择旧集合或单 case；未选择时提示改用连续 source，不静默回退旧四条。
新默认集合目前为 832×480；736 档未验证，仍按下文两档独立验收要求执行。
四条每轮 RAW RGB 约 26.04 GiB，预热一轮加正式三轮约 104.15 GiB。

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
source-to-output 延迟、tail drain、credit wait、播放停顿，以及有 prompt 切换时的
**Prompt-switch TTFF**。有限输入与连续 source 分开比较。保存命令、源码 SHA、配置、预热/正式次数、失败与未执行数、
原始 JSON 和报告；未采到的指标标为未测。

## 固定 case 预热和缓存 TTFF 对照

常驻服务启动后，可用现有固定 case 和正常请求预热 Qwen、参考图预处理及 VAE latent：

```bash
python3 benchmark/zing_h3_causal/bench_streaming_source.py \
  --url ws://127.0.0.1:30000/v1/realtime_video/generate \
  --media-root /path/to/media --case 30s-wasteland \
  --prewarm-only --warmup-runs 1 --no-stage-timing --output-dir prewarm-wasteland
```

`--prewarm-only` 不执行 measured 请求；仍完成正常生成与会话清理，保存预热请求、
错误和输出回执。它不是只跑 Encoder 的专用 RPC，也不缓存 source 视频。
用 `--case all` 按固定顺序预热全部内置 case；预算不足时早期条目可能被淘汰，
因此不能仅凭预热完成判断所有 case 都驻留。预热后保持同一服务进程，再用
`--warmup-runs 0 --measured-runs 3` 测重复请求。

测 Block0 condition cache 时，服务需开启 `--zing-block0-cache-mode on`，
客户端还需传 `--cache-write`，允许完整预热请求填充缓存（默认不写入）。
保持同一服务，用相同数据再跑至少一次，并检查实际 `cached_prefix_hits`、
`cached_prefix_misses` 和恢复记录后，才将后续请求计为 cache hit 测量。
包含 prompt 切换的数据须容纳全部 condition；host 预算是全 SP 总额，
不能把全局预算当作每 rank 的容量。仅开启服务缓存或完成预热不足以证明命中。

缓存对照固定源码、硬件、checkpoint、输入 SHA、有效几何、采样、编译与 SP 布局：

| 组 | 设置 | 记录 |
| --- | --- | --- |
| 关闭新增缓存 | `--zing-reference-cache-bytes 0`，保留原 Qwen 缓存设置 | runtime 预热后 3 次正式请求，作为本改动的主要 baseline |
| 首次填充 | 新服务启用参考图缓存，首次请求 | 单独记录，可能含编译/Graph 首次开销，不混入正式中位数 |
| 重复命中 | 同一服务预热后，同一输入重复 3 次 | TTFF 中位数/P95；核对 reference 统计及 Qwen hit/miss |

若另测所有准备缓存关闭，local Qwen 再设 `--zing-conditioning-local-cache-bytes 0`；
remote Qwen 的 `limits.cache_bytes` 和 `limits.cache_bytes_per_tenant` 都设为 0，
与保留已有 Qwen 缓存的 baseline 分开标注。分辨率 480/736 分组报告。
VAE/SP 部分命中会重算，缓存 `hit` 是每 rank 的本地查询计数，不单独证明 GPU forward
已跳过；结合插入计数和独立阶段诊断判断。cache `current_bytes` 只计驻留 CPU tensor，
不含请求副本、Python 对象、文件读入和临时传输缓冲，不等于进程 RSS 或显存。

## Prompt-switch TTFF（完整帧批次接收口径）

连续 source 入口使用 `bench_streaming_source.py --prompt-updates updates.json`；
清单按 `source_frame_start` 递增安排更新，例如：

```json
[{"source_frame_start": 48, "prompt": "Change the lighting to blue."}]
```

位置表示客户端提交到该 source 帧边界时发送更新，不保证模型在该帧切换。
更新采用唯一 `event_id`；以输出中实际携带该事件的首个帧批次匹配，ACK 不算生效。

```text
Prompt-switch TTFF / ms = 1000 × (
    客户端完整收到首个匹配 event_id 的帧批次 payload 的 monotonic 时间
  − 客户端发送该 prompt 前的 monotonic 时间
)
```

计时包含序列化、发送队列等待、上传、服务端等待/条件编码/生成和输出传输；
排除本地参考图片文件读取、浏览器解码和渲染。缺少后续 source 时的等待也包含在内，
因此 baseline/candidate 必须固定 source 节奏、切换位置和帧批次配置。
同一批次含多帧时须等完整 payload 收到，不能解释为该批第一帧字节到达时间。
该指标证明新条件被采用，不证明画面语义已成功变化；视觉效果须单独验收。

JSON 的正式字段为 `prompt_switch_ttff_ms`（此前名为 `prompt_update_latency_ms`，
计时算法不变）。单次客户端 summary、`prompt-updates.json` 和请求 `streaming`
均保留分布；`summary.json` 的 `streaming.prompt_switch_ttff_ms` 按成功正式会话
中的 applied **事件**汇总，报告中展示中位数/P95/事件样本量及逐事件记录。
样本不是会话数，不平均每会话的中位数或 P95。480、736 及各 case 分开报告。

`prompt_updates` 保留 `event_id`、`submitted_seconds`、`received_seconds`、
`latency_ms`、`output_frame_start` 和 `pts_us`，以及
`applied / rejected / superseded / not_observed / interrupted` 状态。
失败会话与预热不进入正式分布；被覆盖、拒绝或未观察到的更新不记为零延迟。
无更新时分布样本数为 0，不代表零延迟。HTML 保留正式轮的逐事件状态，包括失败会话。
旧报告仍可按逐事件 `latency_ms` 展示相同计时口径。

## Sink 切换回执

连续 source 入口使用 `bench_streaming_source.py --sink-updates sink.json`，可同时传
`--prompt-updates`。例如：

```json
[{"source_frame_start":348,"operation":"off"},{"source_frame_start":480,"operation":"refresh"}]
```

帧位置表示客户端提交 source 的发送边界；实际生效以输出 header 的 `sink_event_id`
为准。部署通过 ready 的 `sink_updates` 声明支持；ACK 的 `accepted` 只表示已接纳。
同一 source 帧可安排多个操作，尚未生效的操作采用 latest-wins。

每次请求目录的 `sink-updates.json` 保存发送位置、实际生效的 `output_frame_start`、
`sink_chunks`、全局 `sink_chunk_ids` 及 `applied / superseded / not_observed / interrupted`
状态；`events.jsonl` 中的输出 `source_alignment` 可用于回查被选中块对应的原视频区间。
失败请求也保留已发送操作的回执。完整行为约定见
[动态 Sink 设计](../../docs/docs/sglang-diffusion/zing/specs/dynamic-window-sink.zh-CN.md)。

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

## 旧有限输入集合：从 checkpoint 一次运行

在已安装 SGLang 的 NVIDIA 环境中运行：

```bash
python3 benchmark/zing_h3_causal/run_benchmark.py --model-path /path/to/sglang-checkpoint --case-set whitebox-basic
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

默认集合为 [`product-case-20261029`](case_sets/product-case-20261029.json)，使用上文连续 source 入口。
旧集合 [`whitebox-basic`](case_sets/whitebox-basic.json) 包含下面由 Xiaoxiao 提供的四组输入，
须显式选择，用于旧有限输入与历史结果回归。
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

默认真实请求集合 [`product-case-20261029`](PRODUCT_CASE_REPLAY.md) 覆盖单图、长
prompt、动态增减 reference 和完整长会话；旧四个 case 的内容保持原样。该集合通过
`bench_streaming_source.py --case-set product-case-20261029` 运行，各 case 自动使用自己的
条件更新序列；不能用于有限输入入口。素材准备、回放边界与验证状态见该说明。

## 旧集合的固定输入与部署

| case | 固定身份 | 输出 | 参考条件 |
|---|---|---|---|
| `5s` 海岸 | `zing-whitebox-video-five-triview-0001` | 120 帧、24 FPS | 六图 |
| `30s-wasteland` 废土 | `zing-whitebox-video-30s-six-triview-0001` | 720 帧、24 FPS | 全局图 + 六张三视图，共七图 |
| `30s-town` 地中海小镇 | `zing-whitebox-video-30s-six-triview-0002` | 720 帧、24 FPS | 全局图 + 六张三视图，共七图 |
| `30s-indoor` 刺绣霓虹漫画／室内通道 | `zing-whitebox-video-30s-two-triview-0001` | 720 帧、24 FPS、832×480 | 全局图 + 主角／林姐三视图，共三图 |

PR #299 的室内用例保留原始三图条件与 prompt；素材及请求准备已验证，GPU 生成效果待实测。
人工检查重点为人物身份、入镜／离镜时序、参考下排外观与长期闪烁／漂移。

2026-10-12 按用户确认采用 PR #261 的两条准确七图长视频输入。
旧有限输入工具显式选择 `--case-set whitebox-basic` 时串行执行四个旧 case；长视频使用
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
python3 benchmark/zing_h3_causal/bench_client.py --case-set whitebox-basic \
  --url ws://127.0.0.1:30000/v1/realtime_video/generate \
  --media-root /data/leap-worldmodel-thailand \
  --sampling-params /data/benchmark/sampling.json \
  --deployment /data/benchmark/deployment.json \
  --output-dir /data/benchmark/run-001 \
  --warmup-runs 1 --measured-runs 3
```

单 case 用 `--case 5s`、`--case 30s-wasteland`、`--case 30s-town` 或 `--case 30s-indoor`。
`--preflight-only` 核验输入并保存配置，不连接服务。输出目录必须是新目录。
源码以冻结归档部署、没有 `.git` 时，显式传 `--source-revision` 的完整 Git SHA；
修改后的客户端另以文件 SHA256 记录，部署方仍须校验归档和覆盖文件。
默认 `--media-mode representative`：每个 case 只为首个正式请求保存代表 MP4，
其余预热/正式重复只留完整指标、帧数/字节数和输出 SHA-256；所有请求仍完整运行。
代表视频从全程均匀选择最多 120 帧，最大宽度 640，以 12 FPS 编成 time-lapse；
抽样原图内存上限 128 MiB，视频上限 64 MiB，FFmpeg 编码在请求结束后进行。
`media.json` 保存抽帧索引、全量 RGB 哈希与媒体路径。它不能替代完整成片或像素 parity；
质量验证需按动态换图等冻结边界另保留所需证据。直接调用连续客户端而未提供总帧数时，
代表视频仅采样最初的有界帧，不能称为全程覆盖。

`--media-mode metrics` 完全不留视频；`--media-mode raw`（有限客户端也支持旧的显式
`--save-media`）才写原尺寸 RGB，并受留存门槛限制。新模式仍逐 payload 验证尺寸、
格式、字节数和顺序，保存完整流的 SHA-256。有限接口的输入哈希覆盖序列化 init，
其引用资产另有独立哈希；连续接口输入哈希覆盖实际发送的 RGB。

### 实验结束留存与缓存清理

单个实验的所有 case、配置和 attempt **合计不超过 2 GiB**（`2 << 30` bytes）；
统计同时约束文件逻辑大小与已分配块，不跟随共享模型/素材符号链接。
benchmark 写 `retention.json`，超预算报错，不能标记完成，不自动丢弃指标或准确性证据。
`run_benchmark.py` 还核对包含 server 日志和报告的总输出根目录；外层矩阵 launcher
必须用同一个实验根目录再验收，不能把每个 cell 的门槛当成总量通过。

Block0 KV 是运行时可重建数据，不作为每轮长期留存。`run_benchmark.py` 显式使用
`--block0-cache-disk-gb GB` 时，在新输出根目录创建受管缓存并在 server 停止后清理；
这要求冻结 runtime 支持 Block0 参数，不改变未指定时的原缓存配置。
独立 resident 客户端不会清理服务缓存，缓存收尾由拥有 worker 的外层 launcher 执行。
新 launcher 使用冻结工具：

```python
# 示例 API；必须在本次专属实验根目录、worker 启动前取得 lease。
from benchmark_artifacts import ExperimentRetention
retention = ExperimentRetention(experiment_root)
try:
    cache = retention.create_cache("scratch/block0-cache")  # 必须此前不存在
    # 启动本次 server，--zing-block0-cache-dir 指向 cache；执行完整矩阵。
    # finally 中按 PID/启动身份停止本次所有 worker，并核实退出。
    retention.finish(workers_stopped=True)
finally:
    retention.close()
```

`workers_stopped=True` 是调用方已核实退出的前置条件，不能在仅结束客户端请求后设置。
工具只清理本次登记创建的缓存；先保存 JSON manifest/哈希，再删除 KV/conditions/.tmp
payload，写 `cache-cleanup.json` 和留存回执。不接管已有缓存或共享服务缓存；
拒绝并行 owner、路径越界、符号链接、挂载点和目录身份变化。不得在活跃任务根目录
更新脚本，也不得扫描 `/mnt/zing-nvme` 按日期或目录名批删。

成功、请求失败、OOM、超时和 SIGTERM 使用同一个 launcher finally。SIGKILL/节点故障
不能保证自动收尾，遗留缓存需在核实精确 Job/Pod 终态、归属和活跃引用后单独恢复。
必要的完整视频、无损窗口或逐 rank trace 超出预算时，按当前授权归档并校验远端
大小/哈希后释放本地副本；无授权或校验失败则明确留存未通过，不能静默删除。

哈希和有界抽帧在接收路径上，MP4 编码在计时结束后；旧的逐包写 RGB 结果与新模式
须同保存模式重测 baseline。新增默认策略已做 CPU/本地 WebSocket 与媒体检查，
当前 H100/H200/B300 的 480/736 整体性能及三组件耗时/显存**未重新验证**，不声称加速。

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

### 连续 source 的独立输入与输出尺寸

`bench_streaming_source.py` 可保留标准 case 的素材、参考图、prompt 与帧数，显式派生
输出尺寸及 source RGB 尺寸。例如测试 **source 832×480、输出 1280×736**：

```bash
python3 benchmark/zing_h3_causal/bench_streaming_source.py \
  --url ws://127.0.0.1:30000/v1/realtime_video/generate \
  --media-root /data/media --output-dir /data/benchmark/source480-output736 \
  --case 30s-wasteland --width 1280 --height 736 \
  --source-width 832 --source-height 480 \
  --warmup-runs 2 --measured-runs 3 --no-stage-timing
```

本 PR 的客户端已支持独立 source 几何：视频解码、RGB 字节计数、传输声明与 credit
使用 source 尺寸，输出 RGB 校验使用 output 尺寸；raw 文件和管道必须提供声明尺寸的 RGB24。
服务端尺寸解耦仍依赖 [PR #370](https://github.com/seedleap/sglang/pull/370)（2026-10-29仍未合入）；
benchmark 参数不会自行启用服务端尺寸解耦。
服务端必须显式启用 `zing_source_resize={"max_short_edge":480,"max_area":399360}`。
客户端先把视频解码为声明的 source 尺寸，服务端仍按输出 canvas 生成 target noise、
DiT target latent 和 Decoder RGB；不得把 source 缩小后又放大来代替此路径。
本轮快配方同时启用 `zing_reference_resize`，取相同短边和面积上限；两项策略均由部署配置记录。
省略尺寸参数时保持标准 manifest 的尺寸；宽高必须成对指定。

派生 case 保留 `original_case_id`、`original_manifest_dimensions` 和资产身份，另记录
`source_width/source_height`、输出 `width/height` 及含两种 canvas 的新 case ID。
比较结果时核对实际 Encoder shape、target latent shape 和 Decoder 输出，不仅检查 CLI。
全程 FPS 与首包后 FPS仍按输出帧统计。Source VAE、DiT、独立 Decoder 的实际 CUDA
执行区间应另采诊断；调用方提交 remote decode 的 Host scope 不能当作 Decoder 耗时。
