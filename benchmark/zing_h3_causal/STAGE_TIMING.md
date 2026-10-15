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
