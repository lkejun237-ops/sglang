# Zing H3 组件 FPS 与延迟

2026-11-12：新增 Qwen VLM、Encoder（Source / Reference VAE）、DiT、Lynn
Decoder 的 Prometheus 观测和 Grafana 配置。CPU 合约验证与线上上线验收分开；
已提供采集器微基准；四组件真实模型的 GPU 同条件性能/显存对照及线上接入仍未验收。

## 最短使用路径

先部署包含本 PR 源码的模型进程，再发一次正常请求。新增采集默认开启，
`SGLANG_ZING_COMPONENT_METRICS=0` 可关闭；不需要单独启动 exporter。
未执行过的组件/标签通常没有样本；cache 命中而跳过计算时，不伪造耗时或 FPS。

独立三角色的默认管理 HTTP 端口如下（以实际启动参数为准）：

| 进程 | 默认 `/metrics` 地址 | 来源 |
|---|---|---|
| Encoder（含内嵌 Qwen） | `http://ENCODER_HOST:31000/metrics` | 默认输入端口 30000 + 1000；`--http-port` 可覆盖 |
| DiT leader | `http://DIT_HOST:33000/metrics` | `--http-port` 默认 33000 |
| Decoder | `http://DECODER_HOST:36000/metrics` | `--http-port` 默认 36000 |
| 独立 remote Qwen | `http://QWEN_HOST:8093/metrics` | conditioning `--port` 默认 8093；需要现有 Bearer 凭据 |

例如在 DiT 所在机器、使用默认管理端口时：

```bash
curl --fail --silent http://127.0.0.1:33000/metrics | rg '^sglang_zing_component_'
```

默认进程本地 registry 中，DiT 端点只包含该进程的指标；不能指望它同时返回远端
Encoder/Decoder。每个角色各抓取一个管理端点；内嵌 Qwen 随 Encoder 一起抓取。
多 service 共用一个模型进程时只抓一个，避免重复计数。集成父/子进程的配置见下方抓取说明。

### 部署自动配置

`deploy/sglang/launch.py run` 自动设置角色 `service`、默认 `lane="default"`，
并在导入 Prometheus 前创建按 worker boot 隔离的采集目录。
角色配置可指定 `environment: {"WORLD_MODEL_METRIC_LANE": "test-copy"}`；
service 与目录由启动器管理，不接受角色配置覆盖。
从 `receipt.json.metrics.url` 获取抓取地址；该配置记录不代表已产生样本或已接入 Grafana。

Whitebox 的 H100/H200 整组 supervisor 和 B300 supervisor 同样为每个角色、每次重启
分配新目录、设置正确 service；独立 conditioning 的 service 为 `qwen_vlm`，
内嵌 Qwen 随 Encoder。原生 independent launcher 未收到显式目录时也自动分配；
直接调用时若显式传 `PROMETHEUS_MULTIPROC_DIR`，调用方负责先创建空目录并保证
仅本次角色的 ranks 共享。不要让不同角色/实例共用，也不要清空活跃目录。
自动创建的目录保留到运行环境清理，回收前必须确认所属进程及其后代全部退出。

这些改动不修改已运行或 `reuse` 的角色，不创建 Prometheus 抓取任务。
`region/pool` 仍由抓取侧配置，`instance` 通常由 Prometheus 自动添加。
线上发布、抓取端网络和 Grafana 数据验收仍需分别执行。

Prometheus 的最小配置示例（地址是占位符，替换为实际模型管理地址）：

```yaml
scrape_configs:
  - job_name: zing-h3-components
    scrape_interval: 15s
    static_configs:
      - targets: ['ENCODER_HOST:31000', 'DIT_HOST:33000', 'DECODER_HOST:36000']
        labels:
          region: cn-shanghai
          pool: zing-demo-shanghai-01
  # 只在 Qwen 独立运行时增加；不要重复抓取内嵌 Qwen。
  - job_name: zing-h3-remote-qwen
    scrape_interval: 15s
    authorization:
      type: Bearer
      credentials_file: /run/secrets/qwen-metrics-token
    static_configs:
      - targets: ['QWEN_HOST:8093']
        labels:
          region: cn-shanghai
          pool: zing-demo-shanghai-01
```

凭据文件存放现有有效服务 token，限制文件访问权限，不把 token 写进配置或命令行。
导入本文下方链接的 Grafana JSON，选择地域、pool 和 datasource，即可查看曲线。

## 指标与口径

| 指标 | 单位 | 含义 |
|---|---|---|
| `sglang_zing_component_duration_seconds` | seconds | 成功调用的 Histogram；Grafana 乘 1000 显示 ms，提供均值、P50、P95 |
| `sglang_zing_component_frames_total` | pixel frames | 与同一计时样本配对的计算帧数；Decoder 包含 padding；用于处理 FPS |
| `sglang_zing_component_completed_frames_total` | pixel frames | 成功返回的有效处理帧数；Decoder 在裁剪后的本地输出/独立服务交付边界累计；不受 CUDA 计时丢样影响，用于吞吐 FPS |
| `sglang_zing_component_errors_total` | calls | 失败或取消的组件调用 |
| `sglang_zing_component_timing_dropped_total` | samples | 队列满、CUDA event 失败、120 秒未完成造成的 CUDA 计时丢样 |

前两个指标标签为 `component`、`operation`、`service`、`lane`、`rank`、
`resolution`、`measurement`。没有 prompt、session/generation ID 或参考图身份。
`completed_frames_total` 使用相同标签但没有 `measurement`；errors 只有 `component`，
dropped 使用 `component` / `reason`。地域和模型池由采集端添加 `region` / `pool` 标签。
Histogram 实际导出 `_bucket`、`_sum`、`_count`；客户端还可能自动导出 `*_created` 时间戳。

| component / operation | 执行边界 | FPS 分子 |
|---|---|---|
| `qwen_vlm / forward` | `encode_ids` 前向，包含其内部 prefix reuse；不包含前置 processor、RPC 或队列等待 | 实际采样媒体帧数，图片算 1，视频不计 temporal padding；纯文本为 0 |
| `encoder / source` | progressive Source VAE 的 cache.encode 和编码策略；完整 source 的 encode_visual 编码与 posterior sampling | 实际输入 RGB 帧数，不计尾部 padding |
| `encoder / reference` | 未命中参考图 VAE cache 的 encode_visual 编码与 posterior sampling | 实际参考图/视频帧数；cache 命中跳过计算时无样本 |
| `dit / forward` | SourceReadySession 的 chunk 执行，包括 source/target、denoise、commit；drain 包含该迭代的准备与 lookahead | 已提交结果 provenance 中的有效像素帧数；缺少 provenance 的离线 fixture 为 0，不从 latent 数估算 |
| `decoder / forward` | LynnStream.push 中逐 latent decode_step；不含后续 RGB clamp、D2H、发布、队列或播放 | 计算帧包括 EOS padding；有效处理帧在 frame_limit / SourceLynnDecoder / 独立 Decoder 服务的裁剪边界另计，不含 padding |

`measurement="host"` 是调用返回时间；`measurement="cuda"` 是当前设备、当前 stream
两个 CUDA event 之间的 elapsed time，包括 stream 内依赖等待，不是独占 kernel-work。
Host 返回不代表 GPU 完成。每个进程的后台线程仅 query 已完成的 event，再读取 elapsed time；
不调用 synchronize，不写逐调用文件，不保留张量。队列上限 128，满时只丢观测。
CUDA 计时不可用时保留 host 数据，**不把 host 数值填充为 CUDA 时间**。

处理 FPS = `rate(frames_total) / rate(duration_seconds_sum)`，使用相同标签和计时样本。
吞吐 FPS = `rate(completed_frames_total)`，包含空闲和输入不足造成的影响，独立于 CUDA 计时样本。
除 Decoder 在有效帧交付边界上报外，其余组件在函数成功返回时累计；这是逻辑处理进度，
不额外同步 GPU，也不保证帧已传到浏览器。CUDA 样本及其配对计算帧数在后台确认事件完成时更新。
它们都不是客户端播放 FPS。Qwen FPS 是媒体帧处理速度，不能与生成视频 FPS 当成同一业务指标。
不平均每次调用的 FPS，也不固定按 17 帧换算 chunk，不累加重叠组件时间当作 E2E。

`resolution` 使用当前实际像素宽×高，例如 `832x480`、`1280x736`，不合并统计。
Qwen 的多参考输入使用 `conditioning`；没有像素几何的离线 DiT fixture 使用 `unknown`。
独立 DiT 的几何来自已验证的 init，集成 source 路径来自实际 latent 几何和加载的 VAE ratio。
SP 的每个 rank 独立上报，看板仅选择 rank 0，避免重复计数或将 rank 时间求和。
跨 chunk joint 的 lookahead 时间归入发起它的迭代，不能解释为单独某个 chunk 的纯去噪时间。
启动预热也会产生样本；选择预热结束后的时间区间进行稳定态排障。

## `/metrics` 数值示例与查询

以下是**说明用样例，不是线上或 GPU 实测结果**，为简洁省略了其余 histogram 桶和 `*_created`：

```prometheus
sglang_zing_component_duration_seconds_bucket{component="dit",operation="forward",service="dit",lane="default",rank="0",resolution="832x480",measurement="cuda",le="0.5"} 10
sglang_zing_component_duration_seconds_bucket{component="dit",operation="forward",service="dit",lane="default",rank="0",resolution="832x480",measurement="cuda",le="+Inf"} 10
sglang_zing_component_duration_seconds_sum{component="dit",operation="forward",service="dit",lane="default",rank="0",resolution="832x480",measurement="cuda"} 4
sglang_zing_component_duration_seconds_count{component="dit",operation="forward",service="dit",lane="default",rank="0",resolution="832x480",measurement="cuda"} 10
sglang_zing_component_frames_total{component="dit",operation="forward",service="dit",lane="default",rank="0",resolution="832x480",measurement="cuda"} 170
sglang_zing_component_completed_frames_total{component="dit",operation="forward",service="dit",lane="default",rank="0",resolution="832x480"} 170
sglang_zing_component_errors_total{component="dit"} 1
sglang_zing_component_timing_dropped_total{component="dit",reason="queue_full"} 2
```

这表示：10 个成功计时调用累计 4 秒、配对 170 帧，累计平均调用延迟为 **400 ms**，
处理 FPS 为 **42.5**。errors 与 dropped 独立累计，不能拿它们当成功调用样本。
单次 Counter 快照不能推出吞吐 FPS；例如两个相隔 10 秒的快照中 completed 从 0 增到 170，
这段时间的有效处理吞吐才是 17 FPS。一个窗口内已经发生的 dropped 也可能属于其他调用，
不能仅凭上面一个快照推导所有计时的覆盖率。

下面用 5 分钟窗口查询单个 DiT/rank/分辨率；部署中再按实际需要加 `instance`、`region`、`pool`
筛选，聚合多个进程时保留这些维度，不能把不同模型实例时间相加当作单实例处理速度：

```promql
# 平均每次调用延迟，ms
1000 * rate(sglang_zing_component_duration_seconds_sum{component="dit",rank="0",resolution="832x480",measurement="cuda"}[5m])
  / rate(sglang_zing_component_duration_seconds_count{component="dit",rank="0",resolution="832x480",measurement="cuda"}[5m])

# 处理 FPS：CUDA 配对的计算帧 / CUDA elapsed
rate(sglang_zing_component_frames_total{component="dit",rank="0",resolution="832x480",measurement="cuda"}[5m])
  / rate(sglang_zing_component_duration_seconds_sum{component="dit",rank="0",resolution="832x480",measurement="cuda"}[5m])

# 有效处理吞吐 FPS：不依赖 measurement，也不受计时丢样影响
rate(sglang_zing_component_completed_frames_total{component="dit",rank="0",resolution="832x480"}[5m])
```

## 抓取与看板

独立 Encoder、DiT、Decoder 在已有健康检查 HTTP 端口提供 `GET /metrics`；
端口与模型通信端口不同，不新增监听端口。多个 service 共用一个模型进程时，
这些健康检查端点返回相同进程的 registry，**每个模型进程只抓取一个端点**。
多个 rank 的指标仅通过显式配置的 multiprocess registry 汇总，不能假设 rank 0 registry 包含其余 rank。

独立 Qwen conditioning 服务提供 `GET /metrics`，沿用现有 Bearer 鉴权，
抓取端使用已有服务凭据的受限文件，不在配置或仓库里写 token。
Qwen 内嵌在 Encoder 时由 Encoder 进程一起导出；远端 Qwen 必须单独抓取其服务端点。

对于 HTTP 父进程和模型子进程分离的集成部署，在**启动 Python 前**设置
`PROMETHEUS_MULTIPROC_DIR` 为它们共同可写的同一目录；每个部署实例独占目录，
完整重启实例前由启动器清空，不能跨实例或跨重启复用旧计数文件。
`critical_path_metrics.prometheus_latest()` 在该配置存在时用 MultiProcessCollector 导出，
没有配置时保持现有进程本地 registry 行为。不要在活跃实例中清空目录。
仅升级 HTTP 父进程而没有升级模型子进程不会得到新指标。

采集器需要保留 `sglang_zing_component_*` 指标及上述标签、Histogram 的 bucket/count/sum
和 Counter，并添加正确的地域/模型池。已有 Room critical-path 指标不是这些组件指标的别名。
集成 GPU 子进程没有共享 multiprocess 目录时，现有 `/metrics` 的 200 不能证明组件数据已可抓取。

可导入看板：[zing-h3-components.json](../../../../../examples/monitoring/grafana/dashboards/json/zing-h3-components.json)。
导入到 `world-model-data / SGLang` 目录，选择以下数据源与模型池：

| 环境 | datasource UID | region | pool |
|---|---|---|---|
| 上海正式 | `whitebox-sh-native` | `cn-shanghai` | `zing-demo-shanghai-01` |
| 泰国正式 | `whitebox-th-native` | `ap-southeast-7` | `zing-demo-thailand-01` |
| 上海测试副本 | `whitebox-sh-native` | `cn-shanghai` | `test-shanghai-copy` |

2026-11-12 只读查询：沪/泰两个数据源中 `sglang_zing_component_.*` 均返回空结果，
当前 Service Account 在上海正式看板 `canEdit=false / canSave=false`。
本次没有修改线上看板、采集器或重载模型服务。不能把导入文件已生成当作线上已接入。

## 验证与上线验收

CPU 用例覆盖秒/帧单位、异步完成前后计数、无 CUDA synchronize、失败和取消不计成功、
有界队列/超时丢样、关闭采集、HTTP 导出、Qwen 鉴权和视频 padding 计数。
相关 source-session、conditioning 和 visual encoding 回归一起运行。
完整 Lynn 模型 CPU 回归需要包含 `autoencoder_kl_minimax_h3` 的 diffusers；本机环境缺少该模块，
指标合约和短尾测试使用真实 stream/服务逻辑及小型张量计算替身，不代表完整 Decoder 模型验收。
短尾回归同时覆盖有限流、本地 source、逐 latent source 和独立 Decoder 服务：2 个 latent 的
5 个计算帧与 2 个有效处理帧分别计数；队列满/超时/事件失败时有效帧仍完整保留。

上线前须按部署顺序升级模型进程、配置抓取，再在各环境查询真实新样本并导入看板。
由于 conditioning profile 的代码身份包含 encoder.py，远端 Qwen 部署必须重新生成匹配的
profile 并同步客户端期望身份，不能继续使用旧 profile 或跳过身份校验。
还需在同条件下覆盖 480 和 736，分别验证真实帧数、rank、geometry、延迟边界和采集开销；
按 CLIENT_BENCHMARK.md 补齐整体及三组件的性能/显存对照。当前没有这些 GPU 数据，
不声明零开销、吞吐收益或线上可见性通过。可用 `SGLANG_ZING_COMPONENT_METRICS=0`
关闭新增 event/观测，供同条件开销对照；原有 critical-path 指标不受此开关影响。
