# Qwen 提前编码与 NCCL：四卡 W6 单次流程验证

2026-10-28，PR #452，受测提交 `6421f086105d086c835b9d5dcdf7bf7e55b6492c`。
该证据早于随后合并 main `5e66d5e2c3`；不代表合并后的 GPU 验证。

## 配置与范围

- 上海 H 卡资源，设备报告 **NVIDIA L20Z / SM90 / 81,559 MiB**，不能作为 H200 或 B300 数据。
- 四卡：DiT SP2 两卡，Qwen + Source Encoder 共一卡，独立 Decoder 一卡。
- step2500，832×480，hiker 输入 360 帧；W6/sink3、block_relative/gap12、after_step0，前三块四步，随后两步。
- weights-free 静态 AdaLN、FA4、stable shapes、SP CUDA Graph、progressive Encoder；单会话 KV pool，prefix capacity 9728。
- Qwen prefetch、Encoder overlap/prefetch、Worker async output、Decoder overlap、work quantum 1、cross_chunk、defer-final、prompt-joint 全部配置开启。
- Qwen/DiT 配置均省略 `result_transport`，验证部署默认 NCCL。
- 按本次测试范围仅跑一次完整流程，不另跑预热、A/B、736 或重复统计。客户端不按 24 FPS 节流，仍受 transport credit 背压；24 FPS 是视频时间轴。
- 收到至少 120 帧时发送一次 prompt/reference 更新，实际在收到 128 帧时发送，event_id=189。

输入 SHA256：`16e406dfb36239beee9234fdfc7ef698a5efeb31f89039299368bb7e8da40fc0`。
输出原始 RGB SHA256：`94f8a00e6502cf466f19a9b2ee87d247b64a00fcaecb1f843ef61e934789a241`。

## 结果与边界

作业成功退出；提交、接收、消费均为 **360 帧**，`completed=true`，生成完整视频。
后台线程 `zing-qwen-prefetch` 的候选被消费：`hits=1, encodes=1, failures=0, fallbacks=0`。
实际调用 `NcclSender` / `NcclReceiver`，两个 DiT rank 均采用 CUDA conditioning 张量。
更新结果为 `context=[5413,5120] BF16`、`row_role=[5413] I64`，总计 55,472,424 字节。

以下是单次更新的 **Host 调用边界**，同机 monotonic 时钟，无额外 CUDA 同步；不是 CUDA kernel 耗时或多样本 P50/P95。

| 指标 | 实测 |
| --- | ---: |
| 提前编码开始 → Qwen encode_device 开始 | 58.28 ms |
| Qwen encode_device 调用 | 482.37 ms |
| 提前编码全调用 | 543.47 ms |
| 提前编码启动 → 消费方进入 | 150.88 ms |
| 消费方等待候选 | 392.69 ms |
| NCCL send 调用 | 0.72 ms |
| NCCL 发送开始 → 接收调用返回 | 1.55 ms |
| 更新发送完成 → 首个新 event RGB header | 2113.31 ms |

提前编码与 NCCL 可以一起工作，但本次提前量不足，仍有约 393 ms 暴露等待。
1.55 ms 包含 Host/控制开销，不是纯 NCCL GPU 传输耗时；新 event header 不证明画面语义已正确切换。
DiT `_forward` 观察器未产出事件，因此不据此声称精确的 GPU kernel overlap 或 A/B 加速。

运行计数为 `cross_chunk=42`、`prompt_joint=1`，有 deferred/resume 图快照。
六分支先按 `PREFIX_PHASES` 打包成一个 joint forward；路由 `prefix_separate.warmup=1`
表示该联合 forward 的首次普通执行，尚未 capture/replay CUDA Graph，不能将标签误解为六分支拆开执行。
Encoder 已预取至 360 帧、队列曾有 pending results；Decoder 独立 worker 完成输出。
本次没有分别测量 Encoder/Decoder 的 GPU 重叠收益，也未逐项验收所有底层 kernel 优化。

| 整体交付指标 | 实测 |
| --- | ---: |
| TTFF（客户端启动 → 首批 RGB） | 53.718 s |
| E2E（客户端启动 → 最后 RGB） | 74.833 s |
| 全程交付 FPS | 4.81 |
| 首批后交付 FPS | 17.00 |
| TPOC | 0.491 s |
| 输出批间隔中位数 / P95，43 个间隔 | 0.519 / 1.031 s |
| tail drain | 4.421 s |
| credit wait 累计 | 66.035 s |

这是冷流程，首个 dispatch 的 DiT Host 阶段占 46.934 s，包含首次执行/编译等，未继续拆分。
首批后仍含后续 shape 预热、一次更新和尾部排空；两种 FPS 均不代表充分预热后的稳态吞吐。

1 Hz driver 显存采样峰值：DiT 两 rank **73,069 / 71,697 MiB**，
Qwen + Encoder 共卡 **53,891 MiB**，Decoder **12,685 MiB**。
这是整卡采样峰值，不是组件独占显存或瞬时绝对峰值。

未验证：736、同条件 A/B、GPU 张量等价性、重复统计、真实播放端时间、完整生命周期故障恢复。
本轮不作性能提升或全配置性能验收结论。

## 原始证据

- 实验标识：`pr452-h100-w6-20261028-r1`，成功作业 `pr452-h100-w6-gpu-r1d`。
- OSS 归档：`oss://leap-world-shanghai/world-model/sglang/evals/pr452-h100-w6-20261028-r1/readback.tgz`。
- 本地：`D:/pr452-h100-w6-20261028-r1/readback/results/`。
- `formal-480/summary.json`、`events.jsonl`：交付、输入与事件时间。
- `proof-*.jsonl`：提前编码、NCCL、消费方时间与张量元信息。
- `server-480.log`、`environment-480.json`、各 `*.command.json`：实际配置、计数与运行路径。
- `memory.jsonl`：逐卡显存；`formal-480.mp4`：视频；`DONE.json`：完成回执。
- 四次先前 harness 启动失败单独保留，不计为产品回归通过；最终产品源码未加运行时修补。
