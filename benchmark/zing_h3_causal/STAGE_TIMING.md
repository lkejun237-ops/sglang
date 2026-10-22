# Zing 综合 benchmark：计时、算子与报告

统一命令索引见 [README](README.md#综合-benchmark-入口)。本页说明完整请求和 DiT 组件计时，
运行时分配历史见 [MEMORY_PROFILING.md](MEMORY_PROFILING.md)。所有模式复用同一个报告入口，
可将 `--run`、`--include-results` 和 `--memory-run` 的数据放在同一网页切换。

## 全50层完整链路实测（2026-10-19）

本轮运行真实 source、参考图、在线 Qwen、Reference/Source VAE、全50层 DiT 和
**Lynn decoder**；BF16、SP2，第三张卡承载远程 Qwen 和独立 Source Encoder。
每场景1次预热、3次正式请求，window/sink=6/3，四步 `[1000,750,500,250]`，
`joint_forward=off`，weights-free AdaLN table；无预制 conditioning、缩层、offload 或 CUDA Graph。
它与下方 K=1/SP1 组件实验是不同的测量，不能混合比较。

| 测试目标 / 实际设备 | 场景 | TTFF 中位数 / s | E2E 中位数 / s | TTFF 后 FPS 中位数 |
|---|---|---:|---:|---:|
| H20 / H20 96GB | 5s | 15.2109 | 46.6953 | 3.7796 |
| H100目标 / L20Z 80GB | 5s | 5.0961 | 16.0813 | 10.7304 |
| B300目标 / L20D 267GiB | 5s | 4.6155 | 15.1686 | 11.2534 |
| H20 / H20 96GB | 30s-wasteland | 17.7678 | 235.4766 | 3.3028 |
| B300目标 / L20D 267GiB | 30s-wasteland | 5.1592 | 73.5349 | 10.5316 |
| H20 / H20 96GB | 30s-town | 18.1374 | 236.6323 | 3.2906 |
| B300目标 / L20D 267GiB | 30s-town | 5.1598 | 72.9574 | 10.6051 |

L20Z 的两个七参考长例未通过保守内存准入，因此没有提交执行，也没有观察到其 OOM。
H20 的两个30s使用 r3 完整客户端/StageTrace 数据；最新5s与 phase/profile 使用 r5。
r3 的客户端测量完成后，独立 phase 断言因遗漏生产 `step` 路由而失败；r5 同时覆盖
`step` 和 `drain_ready` 后通过，不用缺失的 r3 phase 填值。

客户端从连接和启动 source 解码前计时，到最后一个有效 RGB batch；另存
`init_send_to_rgb`，对应序列化 init 后、WebSocket send 前到解包前的 RGB 到达时间，
便于对照旧链路。FPS 按每次请求的 `(总帧数−首批帧数)/(E2E−TTFF)` 计算后取中位数，
不是两个中位数相减。原始 `fps` 保持全程口径，新值为 `post_ttff_fps`。
客户端和服务同节点，通过回环 WebSocket 接收 RGB，未计播放器显示延迟。

| 实际设备 | 3-latent CUDA / ms | 2-latent CUDA / ms | 配对3+2 CUDA / ms |
|---|---:|---:|---:|
| H20 | 2300.616 | 1582.597 | 3883.354 |
| L20Z | 654.359 | 489.917 | 1145.976 |
| L20D | 543.620 | 531.928 | 1078.188 |

上表取5s内独立生产 step 的 chunk9/10，同次请求先求3+2之和，再统计3次请求。
每个 phase 包括 source、4步 denoise 和 clean commit；只记录异步 CUDA events，
在结束处收集，不逐调用同步。3+2 不含中间 Decoder；这是 CUDA stream elapsed，
不能与下方 K=1 的同步 wall baseline 当作相同指标。

算子图使用另一次诊断 profile：rank0 chunk9/10 **及中间 Decoder** 的独占 kernel-work ms，
分类为 GEMM、Attention、KV/RoPE、Other，第三卡的 Qwen/Encoder 不进入这个饼图。
全链路阶段饼图则按 Qwen、Reference VAE、Source Encoder、DiT、Decoder 的累计
CUDA 时间中位数归一化；并行阶段之和不是客户端 E2E。

B300目标原在线算子检查覆盖率93.45%，因此原 `GPU_COMPLETE.passed=false` 保留。
离线从同一原始 trace 的唯一 launch correlation 补充识别两个异步 NCCL SendRecv，
仅归入 Other/Communication，共42.649501 ms；补充归因后覆盖率100%，
总计651.140675 ms、34,334 kernels 守恒，95%门槛未降低，无GPU重跑。
结果 JSON 同时保留原始失败状态和离线归因回执。

### 重建网页与数据来源

[结果 JSON](results/issue334-full-chain-20261019.json) 保留21次正式请求的客户端数据、
组件累计值、原始 phase calls、算子计数/时间、原始运行状态和归档 SHA256。
可直接从原网页“打开结果 JSON”导入，或无GPU重建同一报告布局：

```bash
python3 benchmark/zing_h3_causal/build_benchmark_report.py \
  --include-results '全50层=benchmark/zing_h3_causal/results/issue334-full-chain-20261019.json' \
  --include-results '旧结果=/path/to/old-report.json' \
  --output /path/to/report.html
```

省略第二个 `--include-results` 可只展示本轮结果。网页随场景更新阶段占比大饼图，
同时保留独立3/2 phase图、算子饼图/绝对毫秒图和旧数据切换。导入时从逐请求
数据重算统计，并检查算子守恒；静态图仅在其数据指纹匹配时展示。

运行使用冻结源码归档 SHA256
`5f8ef1337f5c8adce899156bba61fb8db2901e70f6058850a8cb6ee8f717bee6`
及 PR364 Hopper 内存准备、weights-free AdaLN table 和计时 overlay；不是最新 main
的GPU验收。每份归档的 source/overlay/assets/model-config SHA256 及实际硬件保存在
结果 JSON 的 `provenance`/`status.devices`。完整客户端到达记录、StageTrace、
冻结运行脚本、独立 Chrome trace 和媒体留在私有实验归档，不提交权重或大文件。
本轮记录性能和媒体交付，不替代画质或跨设备数值等价验收。

## DiT phase 组件测速

`bench_dit_phases.py` 在独立 SP1 进程中复用生产 DiT、packer 和 sampler，
测量固定真实 conditioning 下的缩层模型。它不启动服务或执行编码、解码，
也不替代客户端 TTFF/FPS。输入必须是本项目可信流程生成的本地
`H3InferenceInputs` snapshot（含真实 source 和六参考图），不能接受外部上传的 pickle。

没有 snapshot 时，先用已有素材准备流程生成默认
`zing-whitebox-video-five-triview-0001` 的本地 `request.json`，再包住已有 validator
做一次不计时的捕获（会运行真实编码与请求，不属于组件测速）：

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/capture_dit_conditioning.py \
  --output /path/to/conditioning.pt -- \
  python benchmark/zing_h3_causal/validate_real_checkpoint.py sglang \
  --sglang-root . --model /path/to/exported-checkpoint \
  --request /path/to/request.json --output /path/to/capture-run \
  --dit-placement layerwise --sp-size 1 --solver few_step \
  --denoising-step-list 1000 750 500 250 --seed 42 \
  --guidance-scale 1 --warmup-runs 0 --measured-runs 1
```

wrapper 只在该子进程及其 workers 中捕获 `ZingH3Codecs.prepare()` 的真实返回值，
保存 CPU snapshot、请求身份和哈希 sidecar；不修改 sampler，也不作为性能证据。
捕获使用的完整 checkpoint 可通过 layerwise 降低 GPU 权重驻留，仍需按实际资源
预算安排一次准备。三卡测量复用同一个已核验 snapshot，不各自重新编码。
默认用例的素材准备和哈希校验入口沿用 [CLIENT_BENCHMARK.md](CLIENT_BENCHMARK.md)。

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/bench_dit_phases.py \
  --model /path/to/exported-checkpoint --conditioning /path/to/conditioning.pt \
  --output /path/to/new-result-directory --layers 1 \
  --attention-backend fa4 --warmup-runs 1 --measured-runs 3
```

层数在构建前修改；checkpoint iterator 在读取 tensor 前过滤未选 block。
当前只选择 `[0,K)`，每层独立驻留，保留完整宽度、refiner、权重来源和六项参考条件。
固定 832×480、120 帧、video-only，分块 `[3,2]` / split-first，启动 `1,2,2`。
四步采样默认 `[1000,750,500,250]`；window/sink 默认为 6/3，可显式设置。
比较组固定 SP1、resident、无 offload、eager、独立 source/denoise/commit，
不混入 joint-forward 或 Graph 结果。

两种计时各自预热后测量：

- `block`：只在生产 sampler 的完整 chunk 边界同步；包含 source、所有采样步、
  clean commit 和 sampler 开销。稳定 `3+2` cycle 按同一 run 内连续 chunk 的
  原始时间相加后统计，排除启动 `1,2,2` 与不完整尾块。
- `forward`：分别同步计时 prefix、source、每步 denoise 和 clean commit。
  这是隔离调用诊断，会改变原有发射节奏；其求和不能替代上一轮 block baseline。
  不在每层添加同步，不开启 profiler。

`samples.jsonl` 保留所有预热和正式样本，`runs.json` 记录有限值、输出 shape、
forward 次数以及 ready/peak allocation。`summary.json` 排除预热，按
chunk、latent phase、forward 类型、step 和 prefix/history rows 分组，
给出 wall/CUDA event 的 count、median/min/max，不混合不同 history。
CUDA event 是当前 stream 经过时间，wall 包含完成等待，均不是纯 kernel busy time。
Q/KV 字段是 SP 切分前的逻辑 packed rows，具体 backend 的多分支 kernel shape
仍须从独立 profile 核验。

`config.json` 保存实际设备、CC、UUID、显存、时钟/温度、功率上限与拓扑，
torch/CUDA/NCCL/backend 版本、参数字节、加载峰值、conditioning 哈希和选中权重的
逐 tensor 哈希。RoPE 使用 checkpoint 的 runtime 配置；AdaLN projection cache
固定为开启、128 MiB，FA4 metadata reuse 固定开启，并记录模型实际生效值。
所有重复与两种计时模式的 latent 哈希必须一致；有限值检查和哈希均在计时外。
不同 GPU 另起进程和结果目录；源码、K、权重、history、环境与
计时模式一致后才可计算速度比。本工具不外推完整深度 FPS。

### 离线展示网页

沿用客户端报告的生成入口，`--run` 指向组件测速结果目录：

```bash
python3 benchmark/zing_h3_causal/build_benchmark_report.py \
  --run L20D-K1=/path/to/phase-results \
  --include-results 旧客户端=/path/to/old-report.json \
  --output /path/to/phase-report.html
```

工具读取 `config.json`、`runs.json` 和 `samples.jsonl`，从原始调用重算分组与
配对 cycle，不依赖缓存的 `summary.json`。同时输出可导入的 `phase-report.json`。
网页默认展示末个完整 `3+2` cycle，可切换配置、block/forward、wall/CUDA、
chunk、latent phase 与调用类型，查看逐 step 的样本数、median/min/max 和
Q/prefix/history/KV rows。启动块和尾块可单独选择，不混入稳定 cycle。
页面保留实际硬件、层数、权重与输入来源、逐次输出和显存记录。

多个组件结果可重复传入 `--run`，在网页中切换配置；`--include-results NAME=JSON`
可重复传入已有客户端或组件报告，将其中所有结果直接嵌入同一网页，打开后即可切换。
首屏显示周期与整块耗时，以及各步调用的 3/2 phase 对比；完整样本、history、
波动与计时口径收在下方详情。也可以继续通过“打开结果 JSON”导入，或导出当前结果。
网页和 JSON 均可离线使用，不上传原始数据。

### 单设备验证记录（2026-10-17）

PAI 作业 `dlchge617f2tc93s` 通过；请求的 B300 配额实际分配 **NVIDIA L20D / SM103**。
K=1（block 0），完整宽度与 refiner、BF16、SP1、FA4、eager、resident；
832×480 / 120 帧，checkpoint VAE 推导 latent `[36,30,52]`。
torch `2.13.0+cu130` / CUDA 13.0，flash-attn-4 `4.0.0b19`，其余参数使用上述默认值。
每种模式预热 1 次、正式 3 次，8 次有限输出的 SHA256 全部一致。
每次 prefix/source/denoise/clean commit 数为 1/16/64/16。
正式 peak allocated/reserved 为 5.86/6.94 GiB。

下面是末段 chunk 13/14 各 3 个样本的中位数；cycle 先按 run 配对求和。

| 测量 | 3-latent wall / CUDA ms | 2-latent wall / CUDA ms |
|---|---:|---:|
| 整 chunk baseline | 28.258 / 28.235 | 27.820 / 27.794 |
| source 独立 forward | 4.603 / 4.573 | 4.407 / 4.378 |
| denoise step 0 | 4.601 / 4.571 | 4.504 / 4.476 |
| denoise step 1 | 4.783 / 4.756 | 4.625 / 4.599 |
| denoise step 2 | 3.568 / 3.540 | 3.405 / 3.378 |
| denoise step 3 | 3.550 / 3.519 | 3.324 / 3.297 |
| clean commit | 3.817 / 3.791 | 3.602 / 3.565 |

配对 `3+2` cycle：wall median/min/max = 55.934/55.754/57.804 ms，
CUDA median = 55.890 ms。独立 forward 的和不等于整 chunk baseline。
上述 Q rows 分别为 1170/780，prefix 均为 12246；denoise step 0 的 history
为 10530/10920，step 1–3 与 commit 为 8970/8580，source 为 9360/10140。
逻辑 KV = Q + prefix + history；不同 step/history 没有混成一个均值。

452 条原始样本已取回并重算，113 个分组、6 个配对 cycle 与报告完全一致。
原始数据、冻结源码和环境回执保留在私有 CPFS/OSS 的
`world-model/sglang/evals/issue334-phase-20261017-r4`；不提交权重、snapshot 或日志。
冻结基线为 `0dafc687b02cc6e63c1078d91116fc89e04fad7d` 加本工具 patch，
GPU 测试后仅补充文档与 CPU 测试清单。此时尚未测量 H20/H100；后续全层记录见本文开头。

### 三目标对比与独立算子 profile

在 `bench_dit_phases.py` 命令添加 `--profile-cycle`，正式 block/forward 重复完成后，再运行一次
同种子采样。只对最后一个完整稳定 `3+2` cycle 开启 CPU/CUDA profiler；这次重复
没有逐 forward 同步，完成后继续尾块，并核验 latent 输出与本卡正式重复相同。
`profile.json` 和 `cycle.trace.json.gz` 分开保留，不纳入正式 latency 统计。

profile 通过 CUDA launch correlation 把每个 kernel 分配给最内层相关 CPU scope，
互斥分成 GEMM、Attention、KV/RoPE、Other。未关联 kernel 保留在 Other；关联率须
同时覆盖至少 95% 的 kernel 数和累计时间，分类数量与时间必须守恒。每个 phase
还须覆盖 source、四步 active 和 final。饼图展示累计 kernel 时间的类别占比，
不是 wall 占比；独立 profile 的 wall 也不等价于逐 chunk 同步计时。

```bash
python3 benchmark/zing_h3_causal/build_benchmark_report.py \
  --run 'B300 目标 · L20D=/path/to/l20d/results' \
  --run 'H100 目标 · L20Z=/path/to/l20z/results' \
  --run 'H20=/path/to/h20/results' \
  --include-results 旧结果=/path/to/old-report.json \
  --output /path/to/report.html
```

同一源码归档、conditioning、选中权重、三份模型配置、层数/SP/backend/Graph、
kernel routes、采样设置和 history geometry 相同的目标自动组成对比组，显示：

- 3-latent、2-latent 和配对 cycle 的正式 wall 中位数、min/max、H20 相对倍速。
- 3/2 phase 的 source、各步 denoise 与 clean commit，统一坐标尺度。
- 每目标算子占比饼图及各类别绝对 kernel ms 对比。

页面对导入结果重新计算配对统计并检查 profile 守恒。缺少来源信息的旧结果仍可
切换查看，不参与速度比。Torch/CUDA/FA4 版本属于部署目标环境，页面单列展示；
版本不同的结果只能描述部署目标差异，不能归因成纯硬件速度比。


## 2026-10-12 三 case 扩展

当前默认矩阵是海岸 5s（六图）、废土和地中海小镇 30s（七图），身份见
[客户端说明](CLIENT_BENCHMARK.md)。以下旧版 5s 采集说明保留作历史语义参考。
main 的 conditioner 回调路径在本地执行 Qwen，但不产生独立 Qwen Encoder span；
整体 Conditioning host 区间仍包括该回调。本轮 VAE Encoder（参考 + source）、
DiT、Decoder 独立验收，Qwen 时间标为未测，不伪造 CUDA 结果。

```bash
python3 benchmark/zing_h3_causal/analyze_stage_timing.py \
  /path/to/run/gpu/30s-wasteland --case 30s-wasteland \
  --qwen-scope conditioner-unmeasured --output /path/to/stage-validation.json
```

这不是静默放宽：分析器明确要求 Qwen span 为零并记录未测口径；所有 VAE 条件数、
帧/latent 连续性、货号、rank、CUDA 和 health 约束继续严格核验。
旧版实际采集到一个本地 Qwen span 的工件使用 `--qwen-scope local-span`。

# 固定 5s 模型阶段计时

第一轮只覆盖有限视频接口。请求、source、六项参考条件、采样和 Decoder 均保持
客户端基线的定义，30s 暂不运行。当前连续 source.append 与独立 progressive Encoder 已接入；新的用法见下文。

## 部署口径更正

前一轮客户端基线启动了独立 Encoder 服务，但有限接口没有使用该服务：
它在 DiT worker 内调用 codecs.prepare，编码 Qwen 条件、六项参考图和完整 source。
原 Encoder 日志只有 ready，没有 encoder_step。因此旧报告中的 E1+SP2 overlap
描述需要更正为有限接口、本地编码、DiT SP2；原客户端接收指标仍有效。
本轮不启动闲置 Encoder 服务。资源池采用四卡规格，实际模型阶段运行在两张
SP2 卡上，另外两卡空闲。

## 阶段数据回传

客户端可通过 `--receive-stage-timing` 请求本次会话的阶段记录；checkpoint 入口开启
`--stage-timing host` 时自动启用。服务器在正常清理后、关闭 WebSocket 前发送
`profile_result`，包含各 rank 的原始 spans、采集状态与完整性信息。
客户端保存在 `profiles/`，报告构建器优先使用这些回传数据，无需服务端文件权限。
结果网页的 JSON 格式与布局保持不变，详见 [客户端用法](CLIENT_BENCHMARK.md)。

## 采集和事件语义

普通请求默认不采集；`return_stage_timing=true` 开启本请求 Host 采集和回传。
benchmark 默认开启，无需配置计时环境变量。服务自动创建部署临时目录；
`SGLANG_ZING_STAGE_TIMING_DIR` 仅可选指定父目录，服务只删除其独立子目录。
ACK 后删除本请求文件，失败/取消及部署退出也清理。

| stage | 边界 |
|---|---|
| Conditioning | 整体准备，包含媒体读取/处理和下列 Encoder 子区间 |
| Encoder / qwen | Qwen 输入构建、处理器和条件编码 |
| Encoder / reference_vae | 每个参考媒体的 VAE 编码，一份媒体一个 span |
| Encoder / source_vae | 完整 source VAE 编码，一次执行一个 span |
| DiT | 一个 AR block 的初始化、prefix/KV 工作及全部采样步骤 |
| Decoder | 可输出窗口的解码、overlap、裁剪和按 block 拆分 GPU tensor |

记录 session request_id、dispatch_id、rank、device、span_id、输入输出产物编号，
以及 latent/frame 范围。Conditioning 将子 Encoder 的产物组成供 DiT 复用的条件。
DiT 累积 latent 供 Decoder 使用；Decoder.output_blocks 映射到客户端
chunks.jsonl 的 server_request_id。不能将模型 block 与客户端 frame_batch 强行一一对应。
内部请求缺少 dispatch request_id 时，采集器保留 null 并标记 missing，避免诊断
打断生成；这种记录无法与客户端配对，不能作为完整计时验收证据。

host_enter_ns、host_return_ns、host_handoff_ns 使用同节点 CLOCK_MONOTONIC，
同时记录主机和 boot ID。handoff 表示调用方接收 tensor/metadata，**不意味着 GPU
已完成**。未观察实际入队边界，enqueue_ns 保持 null。本轮没有 D2H 完成、
WebSocket send 或浏览器显示打点，也不从跨客户端/服务端的时钟相减推导网络延迟。

以下为历史 CUDA 数据的口径，当前请求只采集 Host。旧 CUDA 模式在当前 device/current stream 上放置计时 event；有界后台采集器用
query 检查完成后取 elapsed_time，不新增 synchronize，不修改依赖或调度。
cuda_stream_ms 是该 stream 两个 event 间的经过时间，可能包含等待、空隙和通信；
不能叫纯 kernel 时间，不保证覆盖所有旁路 stream。它不映射为 host 绝对时间戳。
CUDA 不可用、event 异常、30 秒仍未就绪均显式记录缺项，不回退为 host/GPU 假值。

后台输入队列和待完成 event 集合各最多 512 项；溢出记录 dropped，不能把丢样
运行称为完整采集。health 文件记录 written/dropped/pending/queued；异常会输出
服务端日志，独立验收还必须核对 span 数量和请求覆盖。生成异常保留失败 span。
health 通过同目录临时文件和原子替换发布，避免服务退出时留下截断快照。

## 分析与验收

`analyze_stage_timing.py` 检查两 rank、每请求八项 Encoder、连续 DiT latent 范围、
Decoder 完整 120 帧及与客户端货号的对应关系，并逐 rank 汇总调用时间和 CUDA 区间。
Conditioning 与 Encoder 嵌套，跨 rank、阶段也可能重叠：**不能把这些累计时间相加
当 E2E**。它们用于定位成本和等待；严格 kernel 归因留给独立 profiler。

```bash
python3 benchmark/zing_h3_causal/analyze_stage_timing.py \
  /path/to/run/gpu/cuda --output /path/to/run/stage-validation.json
```

实验关闭/host/CUDA 三组使用同一冻结代码、模型、镜像、输入和采样。各组独立启动
常驻 SP2 服务，完整预热一次，正式三次，串行运行；每次保存原始 chunk、stage
事件和 RGB。对比接收指标与媒体完整性，记录采集开销及输出差异；三个样本不能
证明长期无扰动，不设置未经校准的百分比性能门槛。

## 三部分显存与性能的归因边界

涉及速度或显存的 Zing H3 PR，必须按 [PR 三部分报告格式](CLIENT_BENCHMARK.md#pr-三部分报告格式)
同时报告 Encoder、DiT、Decoder 的耗时与显存，逐 case、分辨率及 rank 对比 baseline/candidate。
Encoder 分开标注 Qwen、reference VAE、source VAE；远程服务的计时须来自对应服务，
不能将 RPC 等待当作远端纯 GPU 耗时。当前工具未采集的部分写明未测、原因及补测计划。

阶段 Host span 可以包含等待，CUDA event 区间也可包含等待/通信；二者分列，
不能互相替代或相加成 E2E。显存统计独立于 span：共享进程的 allocated/reserved
或共享设备的 driver 占用，只证明该进程/设备的占用，不能自动归因到某个组件。
现有 `memory_snapshots` 的采集与限制见 [resident profiling](SP_RESIDENT_PROFILING.md)。
报告必须区分暖机后常驻快照、重置统计后的请求峰值、加载峰值和 lifetime peak；
无法隔离时保留共享范围，不捏造组件独占值，也不将独立峰值相加为整卡峰值。
