# Zing 固定输入客户端基线

第一阶段只测客户端 TTFF、E2E、FPS、TPOC、ICL。不启动服务、不加载模型，
不做数值 parity、服务端三段计时或并发矩阵。参考官方
[serving benchmark](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling#benchmark)
的常驻服务、预热、重复请求及逐请求记录方式；协议差异使此入口独立于文本
benchmark，不将 token 指标或视频任务轮询直接套到 Zing。

## 固定输入与部署

| case | 固定身份 | 输出 | 状态 |
|---|---|---|---|
| 5s | `zing-whitebox-video-five-triview-0001` | 120 帧、24 FPS、六图 | 使用现有 manifest |
| 30s | `zing-whitebox-video-30s-six-reference-0001` | 720 帧、24 FPS、六图 | 等准确 manifest/素材到位 |

默认 `--case all` 要求两组输入均就绪；缺少 30s manifest 时在连接服务前失败。
不得换成 PR #261 的七图输入，不得循环、裁剪或延长 5s 素材。
可用 `--case 5s` 做明确标注的部分基线；不能称两组已完成。

使用完整 source 的有限视频接口 `/v1/realtime_video/generate`，不使用
`source.append`，不模拟实时输入。原始 request 的 source、prompt、六图及其
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

当前先显式加 `--case 5s`；30s 到位后用 `--manifest-30s /path/to/manifest.jsonl`。
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
| FPS | 实际有效帧数 / E2E，包含首次等待 |
| ICL | 相邻 batch 到达间隔；记录原始值、样本量、中位数/P95 |
| TPOC | 首 batch 后平均 ICL；不足两个 batch 为 null |
| IFL | null；没有独立逐帧到达时间，不插值伪造 |

RAW RGB 完整 payload 经检查即为可解码像素，不包含浏览器呈现时间。
有限接口以 close code 1000、reason `generation complete` 表示完成，发生于清理后。
必须同时满足完整帧数、完整 split batch 和正常完成关闭；其他正常关闭原因也失败。
`completion_seconds` 单独记录，绝不加入媒体 E2E。

## 结果与验收

- `config.json`：输入与资产身份、完整 init、采样/部署和客户端版本、计时口径。
- `requests.jsonl`：含预热、正式成功及失败，部分输出指标也保留但不进入成功汇总。
- `chunks.jsonl`：逐请求/batch 到达时间、帧范围/数量、媒体时长及 ICL，可独立复算。
- `summary.json`：两 case 分别统计，不混合；预热排除，失败及未执行数保留。
- 可选 `.rgb`：逐请求原始媒体，包含失败请求已收到的部分。

主线验收：固定输入正确、串行完整运行、指标可复算、失败不消失。
没有 24 FPS、固定提升百分比或零误差门槛。少量运行的 P95 只是描述性统计，
始终带样本量，不宣称稳定尾延迟或生产实时能力。

持久测试只保留公式、计数、边界和固定输入的短合同测试：

```bash
python3 -m pytest -q benchmark/zing_h3_causal/tests/test_bench_client.py
```

开发时临时验证 header/payload 延迟、控制包、断流、超时、慢写盘；不维护新的
模拟服务框架。真实 GPU 两 case 的验收另跑，CPU/模拟流通过不能替代它。
当前 30s 输入缺失，因此完整两 case 业务验收尚未完成。

本阶段的实际运行、验收状态和工件身份见
[阶段性验收记录](CLIENT_BENCHMARK_VALIDATION.md)。
