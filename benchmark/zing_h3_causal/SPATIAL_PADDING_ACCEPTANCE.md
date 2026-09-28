# #196 渐进 encoder 空间 padding：冻结验收计划

初始基线 `b235905dd7f16a387ac9c801340c3368713c46e8`。候选只改变
`MINIMAX_H3_PROGRESSIVE_SPATIAL_PADDING=off|reflect2d`，运行时默认值现为 `reflect2d`（原冻结实验时为 `off`）。
在 session 创建时固定；不更改 encode policy、tile batch 或其他实验开关。

## 实现审计

原生 Zing `2b6d63302efea2cfbfb17cddf9ee6d455f86bd63` (#293) 在完整
VisualVAE 外包 TorchDispatchMode，将 contiguous BCTHW 的纯空间反射降成
BCT×H×W reflection_pad2d。渐进 encoder 必须保留自己的逐层时间 history，
不能改为调用整个 VisualVAE。显式 helper 只覆盖两个渐进 padding 调用点，
比 dispatch wrapper 范围小；对不符合条件的布局、时间 pad、负 pad、其他 mode
保留 F.pad 的原验证与行为，不强制 contiguous 拷贝。

`34c59e20bba4b4329327a655ce0c5306fa2c7111` 是原生当前 checkout 的祖先。
其中 B300 约 22.7% 为后续空间与 temporal-zero 融合的完整 clip 数据，
不适用于本实现；本分支没有引入该融合 kernel。渐进历史不是每次全零。

## 运行前数值门槛（禁止按结果放宽）

- BF16/FP16/FP32 padding 输出的 bytes、shape、dtype 必须完全相同；覆盖
  T=1/2/4/8/10/17、对称/非对称/零 padding、空间/channel stride、时间切片、
  channels-last、非法 reflection 及 temporal/非 reflect 回退。
- 同设备、同 dtype、同 encode policy、同 tile 分配和 batch 的完整 moments 与
  indexed posterior latent 要求 bitwise；同时记录 max abs、relative L2。
  本优化不改变卷积算法或 batch，故没有理由引入非零容差。
- EOS 1–34、跨 clip、不同分包、非连续 source、future perturbation、异常重试、
  reset/dispose、多 session 隔离沿用真实 encoder 验收工具；不能只跑 padding。
- 逐 rank 记录 tile IDs、input shape、encoder ranks、fallback、peak memory、缓存。
  默认 deterministic/BF16、stack=false、auto tiling，SP2 每卡六 tile。

## 测量与媒体门槛

同节点 SP1/SP2；SP4 在实际执行前为 pending。每个 SP 独立进程按 A/B、B/A
顺序运行；每组完整预热 1/8/8 历史状态，正式至少三次、profiler 单独采样。
完整 encoder 计时包括 gather/拼接；报告首 latent、1/2 latent 分组 p50/p95、
encoder 中位耗时、端到端 FPS/输出间隔 p95、峰值显存、缓存平台期。

只用当前 `source.append` 协议；默认案例 `zing-whitebox-video-five-triview-0001`
的真实 source 与六图，832×480、24 FPS、四步、seed42、video-only。
CPU 准备先核验实际六图、source 时长与 SHA。短素材不伪装为连续 10/60 秒源；
主任务明确授权：120 帧真实源确定性重播 2/12 次，形成 10/60 秒技术压力流；
记录原始/派生 SHA、每 120 帧接缝，同一 session 不在接缝 reset。
该结果不代表连续 60 秒自然运动/业务画质覆盖。
冻结源码 commit、镜像 digest、权重/配置/媒体 SHA、硬件 UUID 与软件版本。

完整 encoder 中位时间至少改善 5%；首 latent、端到端时延及输出间隔 p95
不得退化超过 5%；无 OOM、死锁、泄漏。未达门槛维持实验开关。
正式媒体须完整解码并检查规格，A/B RGB bytes 和人工画质分别报告。
原始逐 rank trace 保存在受控运行目录，单独交付 Perfetto 文件及打开验收。

## 状态

本文件保留运行前冻结的验收标准。SP1/SP2 编码与完整服务 ABBA、数值、缓存、媒体解码和 trace 证据见
[验收结果](SPATIAL_PADDING_RESULTS.md)。数值与性能门槛通过；60 秒重复源压力流
的抽帧疑虑经用户于 2026-09-28 人工复核反馈“肉眼看了没有问题”，以用户复核通过记录。
根据用户要求默认开启 reflect2d，显式 off 仍可回退。SP4 未验证。
