# 完整 Block0 准备缓存：H100 MVP 部分验收

2026-10-27：实现和固定 case 功能验证完成；性能及阶段矩阵未完整验收。
冻结运行时代码现由提交 `6e3a0d2d0e` 保存。PR 整理时同步了 main `290d08c797`，
保留其 Qwen prefetch、Decoder 身份及 prompt 边界逻辑；同步后的版本尚未重跑 GPU，
下列 CPU/GPU 数字均对应冻结源码，PR CI 状态以 GitHub Checks 为准。
三组正式请求观测到 TTFF 中位数 **1.940 → 1.684 → 1.565 秒**，对应观测差值 0.375 秒（19.3%）。
**overlap 计时期间 GPU4 出现外部任务占用，以上差值不能作为独占资源下的最终收益。**
随后 early 独立诊断的第一次预热因 CUDA OOM 失败，剩余诊断停止，没有缩减 workload 或停止其他任务。

## 资源冲突与完成范围

有效子集：三组正式对照和 late 独立诊断，共 20 次完整 RGB（12 次计时、8 次预热）全部逐位一致；四个进程组均完成两 rank、50 层 K/V 清空后恢复校验。
early/overlap 的正式请求完整命中，未新增 Qwen RPC；overlap 两 rank 的全部消费点 copy 均已完成。
early/overlap 独立阶段及 allocator 诊断未完成。736 仍未通过原 SP2 容量预检。

外部容器 `attention-probe-runner`（namespace `minwm-realtime`）在实验期间使用物理 GPU0–4。
停止本次实验后，GPU4 仍有外部 `/opt/sglang/bin/python` 进程，约 1954 MiB 和 10230 MiB。
失败的 NCCL 原始错误为 `Cuda failure 2 'out of memory'`，不是缓存输出不一致。
资源查询见 `resource-ownership.json`，正式请求时间窗口审计见 `resource-audit.json`：

| 正式组 | 窗口内 driver 样本数 | GPU4 峰值 MiB | GPU5 峰值 MiB | GPU4−GPU5 中位 MiB |
| --- | --- | --- | --- | --- |
| late | 51 | 73991.0 | 73203.0 | 786.0 |
| early | 43 | 73991.0 | 73203.0 | 786.0 |
| overlap | 42 | 75951.0 | 73203.0 | 2746.0 |

late/early 的 rank 间差值约 786 MiB，overlap 增至 2746 MiB，增加 **1960 MiB**，与外部 Encoder 的占用一致。
显存差值不是速度修正系数；不据此猜测“扣除干扰后”的 TTFF。
本次直接容器执行只做了启动时空闲检查，未取得调度器排他 GPU 分配；后续须取得四张独占卡再补测。

## 代码、输入和执行

- 基线 `df0f677e97a9b4f7968d26a9728e09528b838d8f`；冻结运行时补丁身份 `8adce94e01892802d277fe0e7710947e66a5d27bb3a6196d465c7aa5ad1f10d7`，五份源码摘要见 `PLAN.json`。
- 本报告对应下述冻结源码；单条目缓存的设计和开关见 [Block0 完整准备缓存](../../docs/docs/sglang-diffusion/zing/plans/BLOCK0_CPU_CACHE.md)。多条目和 prompt-switch 缓存不在本轮范围内。
- H100 测试目标：上海节点 `e01-cn-2zd4xi8m505`，实际 NVIDIA L20Z / CC9.0 / 81559 MiB；原始软件与 UUID 见 `results-attempt2/hardware.json`。
- 物理 4/5 为 SP2 DiT，6 为 D1，7 为 Qwen + E1；832×480、真实 120 帧/5 秒 source、六参考图、seed 123、BF16、4→2（前三块四步）、W6/sink3、after_step0、cross_chunk、FA4、离线 AdaLN、Graph 关闭。
- 原 GPU pool 单槽 24160 rows，prefix 上限 12288；每 rank 预留一个 8.203 GiB pinned CPU KV buffer，SP2 合计 16.406 GiB，准备产物另计。未新增 GPU KV 副本。
- 每组两个预热、三个正式请求；阶段/allocator 另启进程采集。预热第二次包含清空和逐层读回，正式请求不带这些校验或 allocator history。
- TTFF 从客户端连接和 source 解码前起算，到完整首个 RGB payload 收到。三组使用相同 early key，不包含旧隔离实验的 packed-prefix 摘要；不能直接与旧实验的重算/恢复数字拼成新消融。

## 正式观测值（性能验收待独占复测）

表内中位数 / 样本 P95。每组 n=3；三样本 P95 不代表生产尾延迟。
late：原 prepare + 消费处恢复；early：整包命中 + 提前串行恢复；overlap：整包命中 + 异步恢复。

| 指标 | late | early | overlap | n/组 |
| --- | --- | --- | --- | --- |
| TTFF (s) | 1.940 / 2.017 | 1.684 / 1.744 | 1.565 / 1.595 | 3 |
| E2E (s) | 10.080 / 12.231 | 9.683 / 9.741 | 9.349 / 9.379 | 3 |
| 全程 FPS | 11.905 / 12.021 | 12.393 / 12.402 | 12.836 / 12.859 | 3 |
| 首包后 FPS | 14.618 / 14.795 | 14.878 / 14.879 | 15.277 / 15.380 | 3 |
| TPOC (s) | 0.543 / 0.681 | 0.533 / 0.533 | 0.519 / 0.521 | 3 |
| ICL (s) | 0.017 / 1.730 | 0.017 / 1.887 | 0.017 / 1.688 | 45 |
| source → 首包 (s) | 1.900 / 1.928 | 1.646 / 1.654 | 1.521 / 1.526 | 3 |
| tail drain (s) | 5.050 / 7.152 | 4.898 / 4.915 | 4.697 / 4.702 | 3 |
| credit wait (s) | 4.076 / 4.083 | 4.142 / 4.161 | 3.993 / 4.019 | 3 |
| 播放停顿累计 (s) | 3.367 / 5.436 | 3.222 / 3.227 | 3.016 / 3.037 | 3 |

## 命中、复制与首帧路径

| 组 | Qwen hit 增量 | miss 增量 | RPC response MB 增量 |
| --- | --- | --- | --- |
| late | 3 | 0 | 201.559 |
| early | 0 | 0 | 0.000 |
| overlap | 0 | 0 | 0.000 |

每组每 rank 三次正式请求均命中完整条目。late 每请求仍返回约 67 MB condition；early/overlap 跳过 prepare 与该 RPC。
下表单位 ms，中位数 / P95。early 的提交 Host 时间包含显式等待；overlap 在实际消费处等待 event。

| rank | 指标 | late | early | overlap |
| --- | --- | --- | --- | --- |
| 0 | lookup_host_ms | 5.804 / 7.326 | 4.044 / 4.111 | 4.081 / 4.470 |
| 0 | restore_submit_host_ms | 1.723 / 1.989 | 151.106 / 151.270 | 2.549 / 3.090 |
| 0 | restore_cuda_ms | 150.113 / 150.214 | 150.399 / 150.541 | 150.438 / 150.546 |
| 0 | restore_lead_host_ms | 1.746 / 2.022 | 692.413 / 693.188 | 546.635 / 566.977 |
| 0 | copy 消费前已完成 | 0/3 | 3/3 | 3/3 |
| 1 | lookup_host_ms | 3.842 / 4.073 | 3.573 / 3.629 | 3.566 / 3.586 |
| 1 | restore_submit_host_ms | 2.337 / 2.416 | 150.920 / 151.043 | 1.868 / 2.086 |
| 1 | restore_cuda_ms | 150.125 / 150.220 | 150.155 / 150.300 | 150.317 / 150.342 |
| 1 | restore_lead_host_ms | 2.347 / 2.439 | 692.266 / 693.118 | 546.582 / 566.915 |
| 1 | copy 消费前已完成 | 0/3 | 3/3 | 3/3 |

overlap 复制约 150 ms，提交到消费约 547 ms，事件证明复制在消费前完成。
它与初始化等独立工作重叠；未据此宣称 source VAE CUDA kernel 与 DMA 的精确重叠比例。
首个 dispatch 的 Host 时间如下；prepare 包含会话和 decoder 初始化，encoder/decoder 是 DiT 侧操作与等待，不能当作远程模型纯计算时间，也不能跨 rank 相加为 TTFF。

| rank | Host 指标，ms | late | early | overlap |
| --- | --- | --- | --- | --- |
| 0 | prepare_host_ms | 851.630 / 864.905 | 685.755 / 686.393 | 539.943 / 561.091 |
| 0 | encoder_host_ms | 3.360 / 3.426 | 3.487 / 3.519 | 3.492 / 3.756 |
| 0 | dit_host_ms | 574.492 / 578.174 | 479.633 / 487.382 | 482.237 / 490.486 |
| 0 | decoder_host_ms | 1.839 / 1.911 | 2.029 / 2.148 | 1.873 / 1.922 |
| 0 | joint_wait_ms | 333.435 / 333.673 | 333.096 / 333.394 | 332.686 / 332.819 |
| 1 | prepare_host_ms | 300.377 / 312.472 | 164.793 / 165.148 | 16.267 / 16.435 |
| 1 | encoder_host_ms | 556.483 / 565.456 | 524.131 / 524.381 | 527.155 / 548.258 |
| 1 | dit_host_ms | 655.079 / 656.660 | 555.569 / 556.807 | 556.460 / 574.877 |
| 1 | decoder_host_ms | 1.867 / 1.959 | 2.083 / 2.193 | 1.939 / 2.010 |
| 1 | joint_wait_ms | 333.435 / 333.673 | 333.096 / 333.394 | 332.686 / 332.819 |

## 已取得的 late 诊断

只有 late 独立诊断取得三个有效样本。collector 完整、已 drain，无丢弃、缺 rank 或异常；early/overlap 该部分留空，不能用正式请求的数字补作独立采集。
下面是逐请求累计 Host 中位数 / P95（ms）；source VAE 是独立 E1，reference VAE 是已有小缓存命中路径，Decoder 仅 DiT 侧 dispatch。Qwen 服务内部纯计算耗时和 D1 内部 CUDA 耗时未测。

| 阶段/组件/进程/rank | late |
| --- | --- |
| Encoder/reference_vae/dit-worker/rank0 | 16.633 / 16.674 |
| DiT/total/dit-worker/rank0 | 6119.165 / 6120.177 |
| Decoder/dedicated_dispatch/dit-worker/rank0 | 27.102 / 29.163 |
| Encoder/source_vae/dedicated_source_encoder/rank0 | 3611.876 / 3618.248 |
| Encoder/reference_vae/dit-worker/rank1 | 16.854 / 16.928 |
| DiT/total/dit-worker/rank1 | 6202.064 / 6242.581 |
| Decoder/dedicated_dispatch/dit-worker/rank1 | 25.914 / 34.953 |

allocator 值为预热后常驻、每请求重置后峰值，单位 GiB，中位数 / P95；DiT 是共享进程范围。Qwen 和 E1 共卡分进程。D1 没有独立 allocator 采集，driver 原始采样保留，但受资源冲突影响不作为完整显存对比。

| 进程 | 指标 | late GiB |
| --- | --- | --- |
| encoder/visible=3/device=0 | resident_allocated | 0.672 / 0.672 |
| encoder/visible=3/device=0 | request_peak_allocated | 7.609 / 7.609 |
| encoder/visible=3/device=0 | resident_reserved | 8.008 / 8.008 |
| encoder/visible=3/device=0 | request_peak_reserved | 8.008 / 8.008 |
| qwen/visible=3/device=0 | resident_allocated | 48.158 / 48.158 |
| qwen/visible=3/device=0 | request_peak_allocated | 48.158 / 48.158 |
| qwen/visible=3/device=0 | resident_reserved | 50.164 / 50.164 |
| qwen/visible=3/device=0 | request_peak_reserved | 50.164 / 50.164 |
| server/visible=0,1/device=0 | resident_allocated | 63.246 / 63.246 |
| server/visible=0,1/device=0 | request_peak_allocated | 63.786 / 63.786 |
| server/visible=0,1/device=0 | resident_reserved | 66.062 / 66.064 |
| server/visible=0,1/device=0 | request_peak_reserved | 66.064 / 66.066 |
| server/visible=0,1/device=1 | resident_allocated | 63.246 / 63.246 |
| server/visible=0,1/device=1 | request_peak_allocated | 63.786 / 63.786 |
| server/visible=0,1/device=1 | resident_reserved | 66.057 / 66.057 |
| server/visible=0,1/device=1 | request_peak_reserved | 66.057 / 66.057 |

## CPU、功能证据与回执

- 远端既有环境、GPU 不可见的两个独立 CPU 进程各 **732 passed / 6 skipped**；新增六项均通过。准备执行 `t-sh071a3nogew16o` 成功；证据见 `cpu-test-evidence.json`、JUnit XML。
- 本地只做 Ruff、格式、isort、diff 与 CI inventory 静态检查；未安装项目运行时或执行本地项目回归。已接入 conditioning CI；GitHub Actions 未运行。Mintlify CLI 不在编辑机，未执行站点级构建验证。
- 20 次有效完整输出为 120×480×832×3 RGB，SHA256 均为 `2baaeee78db9f1080deb490afd502be063103c5c2476c625ea714dafd93110a7`；直接 KV 校验覆盖四个进程组、每组两个 rank 的全部 50 层。不是 30 次或完整六格矩阵通过。
- 首次 GPU 执行 `t-sh071a48yxp8tts` 的预热因实验模型配置仍指向 32161 失败；`t-sh071a5sygvr8cg` 创建私有模型配置副本，仅将 Encoder 端口改成 32261。
- `t-sh071a5uh9639c0` 因 GPU 占用在准入阶段退出，未启动模型。
- 主对照执行 `t-sh071a6eszzrlds` 完成三组正式和 late 诊断，在 early 诊断首次预热 OOM 后退出 1；已有结果未被覆盖。清理后本次容器任务为空，外部任务保持运行。
- 未验证：独占环境性能、early/overlap 独立阶段与显存、736、Graph、多 case 淘汰、prompt-switch 和长期驻留压力。功能结果、初步计时和缺项分开记录在 `acceptance.json`。
- 原始产物：`/mnt/zing-nvme/early-block0-h100-20261027-r1`；OSS 前缀 `oss://leap-world-shanghai/world-model/sglang/evals/early-block0-h100-20261027-r1/`。端口失败见 `evidence-failed1.tgz`，OOM 及已完成子集见 `evidence-failed2.tgz`，资源审计归档见 `evidence-audited.tgz`。
