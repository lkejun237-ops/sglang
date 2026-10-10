# Issue #253 客户端基线：阶段性验收

2026-10-10：固定 5s 六图输入已在真实 GPU 上完成一个预热和三个串行正式请求。
本阶段交付客户端接收计时与可复算记录；30s 输入及人工视频效果验收继续保持待完成。
指标定义与运行入口见 [CLIENT_BENCHMARK.md](CLIENT_BENCHMARK.md)。

## 验收状态

| 要求 | 实际证据 | 状态 |
|---|---|---|
| 固定 source + 六图，不替换业务输入 | 规范 5s manifest；七项素材逐份校验大小和 SHA256 | 5s 通过；30s 等准确六图 manifest |
| 模型常驻、并发 1、全部串行 | 同一服务，一个完整预热，三次正式请求，正常关闭后开始下一次 | 5s 通过 |
| 完整媒体接收 | 每次 120 帧、22 个完整 RGB frame_batch，正常完成关闭 | 5s 三次通过 |
| TTFF/FPS/E2E、TPOC/ICL 可复算 | 独立复算 requests/chunks/summary 中的时间戳和公式 | 5s 三次通过 |
| IFL 不伪造逐帧时刻 | 当前协议只提供 batch 到达时间 | null，未测量 |
| 失败保留且排除正式汇总 | SP1 预热 OOM，保留 69/120 帧；正式请求 0/3，成功汇总为空 | 失败路径已观察 |
| 视频规格和可解码性 | 三份 MP4 均为 832×480、120 帧、24 FPS；ffprobe 和完整解码检查 | 技术通过 |
| 视频效果 | 用户观看成片，判断主体/参考一致性、编辑效果和时序稳定性 | 待人工观看 |

没有添加未经校准的 FPS≥24、固定提升百分比或数值零误差门槛。
此次客户端运行没有 native 对照、服务端分阶段耗时、profiler、远程网络、多并发或长期稳定性证据。
随后完成的模型阶段采集见 [STAGE_TIMING_VALIDATION.md](STAGE_TIMING_VALIDATION.md)。
这些缺项不阻塞提交客户端工具的阶段性 PR；准确 30s payload 阻塞两组固定输入的完整基线。
人工观看阻塞视频效果结论，不阻塞接收计时工具验收。本 PR 仅引用 issue #253，不关闭它。

## 5s 正式指标

| 请求 | TTFF/s | E2E/s | FPS（120/E2E） | TPOC/s |
|---|---:|---:|---:|---:|
| measured-000 | 11.461305 | 20.622084 | 5.819005 | 0.436228 |
| measured-001 | 11.616559 | 20.883291 | 5.746221 | 0.441273 |
| measured-002 | 11.518455 | 20.790510 | 5.771864 | 0.441526 |
| 中位数 | 11.518455 | 20.790510 | 5.771864 | 0.441273 |

预热 E2E 为 182.682386 秒，排除在正式统计之外。三次共 63 个 ICL 样本，
中位数 0.006781 秒、P95 1.513185 秒、最大 1.591361 秒。批次存在连发，
这些间隔不等于逐帧平滑播放；三次请求的分位数也只能作为描述性统计。
E2E 截止最后一个有效媒体 batch，清理后的正常关闭时间另存 completion_seconds。

## 冻结配置与失败范围

- 同节点 loopback WebSocket；有限视频接口、完整历史、原尺寸 RAW RGB、关闭 pacing、无音频。
- seed=123、guidance=1、few_step 四步 `[1000,750,500,250]`、stable decode；
  模型实际块配置 `[1,2,2]`，`split_first_latent=false`。
- 部署描述更正：有限接口实际在 DiT SP2 worker 内执行编码和解码；独立 Encoder
  服务只有 ready 日志，没有 encoder_step，不构成编码与生成 overlap 的证据。
  实际申请四卡，GPU 0/1 执行请求，2 加载了未被请求使用的独立 Encoder，3 空闲。
  三卡规格拒绝记录保留。客户端帧数和计时结果不受此描述更正影响。
- 驱动报告 NVIDIA L20D，每卡 274113 MiB、compute capability 10.3；
  torch_sdpa、speed 配置，模型 offload=false、use_fsdp=false。
- torch 2.13.0+cu130、uvicorn 0.39.0、websockets 15.0.1、msgpack 1.2.2、numpy 2.3.5。
- 先前 SP1 加独立闲置 Encoder 的两卡运行在预热收到 69 帧后因 DiT KV 扩容 OOM，额外申请 1.31 GiB，
  当时剩余约 342 MiB。保留独立失败目录；没有将部分接收指标计为完整 5s 成绩，
  没有据此对比 SP1/SP2 性能。最初 CPU 预检遗漏 VPC 的准备失败也保留。

## 版本与工件

真实运行采用基础源码归档和未提交客户端覆盖文件；PR 提交后的文件 SHA256 须与下列值一致。
此次客户端运行的服务端未修改；同一 PR 后续新增的可选阶段采集另有独立版本与回执。
模型链接既有 `zing176-20260928-r3/inputs/model`，
配置、路径、大小与元信息有回执；本次未重新全内容哈希大权重。

| 身份 | 值 |
|---|---|
| 基础服务/客户端 Git SHA | `171e22b68e279c299152bc4f7813ea4ca28a34ae` |
| 客户端 bench_client.py SHA256 | `a34c712760b966c85e963d5ce410e3ba879c9c492f70e0936c382f15c05c3e4b` |
| 基础源码归档 SHA256 | `a8812b2e4c975b54ac8a59d55de94fb64d2b9012862895721b37d524f63aaf0e` |
| 容器 digest | `sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015` |
| GPU 成功作业 | `dlcycnf9n986rmbc`，Succeeded |
| CPU 成功导出作业 | `dlc39yayzjsuv8na`，Succeeded |
| GPU SP1 失败作业 | `dlc1tp9bt7ef5pro`，Failed |
| evidence-sp2.tar.gz SHA256 | `21c6abdc1f3d128b5594b998db25b3f954cf3f25462df7eb2695f671164d86e4` |
| 每份正式 MP4 SHA256 | `6e8504479184213c7abc5144a2bbaa9f38ca687920c99c51880579862f517388` |

运行目录为 `/cpfs/world-model/sglang/evals/issue253-client-5s-20261010-r1/`。
私有 OSS 工件前缀为
`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/issue253-client-5s-20261010-r1/`。
`evidence-sp2.tar.gz` 包含 config/requests/chunks/summary、部署、硬件、命令、日志、
源码校验和失败记录；`5s-measured-000/001/002.mp4` 为观看工件，下载后 SHA256 已核验。
原 RGB 留在 CPFS。日志和视频存放运行目录，PR 只提交工具、短测试和验收文档。

## 测试与下一阶段

持久测试用于公式、计数、拆分批次、缺失/乱序、失败汇总和固定输入边界。
本地受影响范围实际结果为 **25 passed / 1 skipped**，跳过项需要本机未安装的 ffmpeg；
上述云端媒体检查已单独执行。四个变更文件的仓库 pre-commit 检查全部通过。

```bash
python3 -m pytest -q \
  benchmark/zing_h3_causal/tests/test_bench_client.py \
  benchmark/zing_h3_causal/test_whitebox_video_smoke.py \
  benchmark/zing_h3_causal/test_whitebox_six_reference.py
```

开发时的一次性 loopback 协议检查验证 header/payload 接收打点、控制包、缺帧、
错误关闭、异常断流、超时和串行失败停止；它们不扩展成持久模拟服务框架，
也不替代真实业务验收。

本轮 GPU 和导出作业已结束。下一阶段等待用户观看视频和准确 30s 六图 payload；
30s 到位后先冻结其素材与部署，再按相同常驻、并发 1 口径运行。
