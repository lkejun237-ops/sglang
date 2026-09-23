# Issue #173：自动 source presentation 与内存 S0

## Review 后的契约

调用方只提供 source、prompt 与 references，不再提供 `source_preview`、
`ar-first-frame` 或 CLI 开关。模型现有 `causal=true` 决定 AR 语义。
完整视频请求缺少显式 video presentation 时，准备层复用 `target.source`
URI 并添加普通 `type=video, clip={start:0,end:0}` 描述。已有显式 video
（包括独立 target 首帧）保留其类型、clip 和顺序，不重复注入。
image/noncausal 请求不自动添加首帧；clipped source 需要显式 presentation。

native 固定源码为 `967df2fd539590b4fa88462ddb67276e5f9a8192`。
训练转换器 `_record` 按 source URI 找到 user media，AR 模式为其设置首帧
clip；native inference 消费该描述，native 没有 `ar-first-frame` 字段。
对照 native 时必须保留等价 source video 条件，不能删掉它后声称输入相同。
训练 target URI 是真值；推理仅提供 URI-free 输出规格，不需要未来 target。

流式服务先发送 ready，首次有效 append 的 S0 RGB 通过原 source command
到达 worker，在内存中构造成 `(3,1,H,W)` uint8 presentation 张量。
不编码 PNG、不写首帧文件、不增加第二份 IPC；同一原包继续进入 source VAE，
source credit 只记一次。首帧等待不设新的 17 帧门槛。Qwen video padding
及重复时间戳只存在于 processor 内，不增加 source 帧或 VAE reference。
后续 source 不替换 S0；prompt 更新沿用原媒体，在既有 chunk 边界重新编码。

六图及其六项 visual reference conditions 原样保留。默认六图 case 的当前
因果请求自动带 source Video 1；历史无 preview 结果仅属于其冻结提交。
独立 target 首帧 case 继续显式选择。source-first-frame case 保留独立 ID
`zing-whitebox-video-source-first-frame-six-reference-0001` 便于追踪，无额外开关。
迁移与调用示例见 [SOURCE_REFERENCE_INPUTS.md](SOURCE_REFERENCE_INPUTS.md)。

## 冻结及验收方法

实现：`ecf32bb47ccfc15241d0a32c5242db621b3fe8e4`。
镜像：`sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`。
一 source + 六图沿用相同原始七项素材（5,012,854 bytes），并重新检查哈希。
source 原始 1280×720、120 帧、24 FPS，显式 ffmpeg bilinear 缩放至
832×480 RGB24，再无损编码 FFV1/bgr0；三路使用同一派生输入，不声称恢复原图。

| 输入 | SHA-256 |
| --- | --- |
| 原 source | `6579ebd3f1d80455b71e30f9795aa494bceb615ab1302a812430f1199ec80f85` |
| 完整派生 RGB24 | `d917780f9c0bb002fbd646b77f23d442d73b5a42d840c4c911cef74f5af7adbb` |
| S0 RGB24 | `7e874dcd2dde2467e4201490eefa57c1b405a737a3294af40828d1b868e34ca1` |
| FFV1 文件 | `657434f038bc9b38ed6b6ea562d888646e17125d63da5b5d12083e9377a69f8a` |
| TAEH3 | `af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba` |

沿用实验前冻结的 exact 门槛，不降低精度：三路 pixels/timestamps/token IDs/
grids/tags/RoPE exact；同一 SP 的自动与显式 presentation context、source latent、
初始 noise、每次 re-noise、RNG state、最终 latent 逐元素 exact；输出帧数、PTS
和原始 RGB exact。诊断 observer 只保存实际模型张量，不改采样，不凭相同 seed
假定 noise 相同。固定 S0 改后续 source，要求 condition exact 而输出变化。

processor 检查从实际 adapter/worker/codec 路径捕获内存 S0 和六图；审计器直接
消费这些真实 decoded media，不把 S0 再写 PNG。native 参考使用固定源码和
独立 Transformers 5.5.3；Torch/torchvision 仍为服务的 2.13.0/0.28.0，
因此 `full_native_dependency_match=false`。服务 Transformers 5.12.1 自检的
版本不匹配单独记录，其他检查必须通过；不推广为完整 native 环境或 learned
features 对齐通过。

## 当前结果

本地 CPU override/macOS stub 回归 746 passed；真实 Linux 镜像另有
746 passed（19.96 s），作业 `dlcaoyaru2cabio0` Succeeded。三路真实 processor
和各路独立 native 参考器的 14 项检查通过。

SP1 `dlc37dh56xo8zafm` → SP2 `dlc108dssqe6ga60` 均 Succeeded，严格顺序执行，
最多 2 张 GPU。每个 SP 的 5 项正常生成、3 项诊断生成、10 项生命周期和
6 项边界检查均通过：共 **16 个生成用例、32 个生命周期/边界用例**。
覆盖一帧/多帧首包、不规则分包、完整 120 帧 source + 六图、暂停与 prompt
更新、empty EOS、非法/重复/越序包、首帧前及处理中的取消/断连、会话隔离。
只发一帧后断连的检查保存实际 scheduler witness，而非凭等待时间猜测。

SP1 43 项、SP2 86 项真实张量快照均 exact（共 **129 项**），帧数、PTS
和 RGB exact。固定 S0 仅修改后续 source 的第三路，condition exact、输出不同。
各正常/诊断生成用例逐 rank Qwen 初始化一次，source alignment/PTS 连续，
submitted/accepted/consumed/received 相等；结束时 ready、KV、bridge、decoder、
codec pixels 和 encoder retained 状态清零，并有输入文件删除和 scheduler
释放回执。诊断 observer 额外同步/落盘，不用于性能结论。

### 正常服务指标（120 帧业务用例）

| 指标 | SP1 rank 0 | SP2 rank 0 / rank 1 |
| --- | --- | --- |
| 首帧等待 host ms | 14.62 | 18.43 / 18.43 |
| Qwen 初始化 host ms | 579.02 | 580.78 / 589.59 |
| S0 到首 DiT started host ms | 1054.92 | 1056.66 / 1070.15 |
| allocator peak allocated GiB | 229.450 | 182.378 / 182.417 |
| allocator max reserved GiB | 229.861 | 182.791 / 182.830 |
| 客户端首输出 s | 4.357 | 4.730 |
| 最大 pending 帧数 | 60 | 60 |
| 有效交付 FPS | 5.691 | 5.498 |

各 rank 首次 dispatch 只有 1 帧 source，已产生 DiT started 事件，无新 17 帧
门槛。host 时钟不是 CUDA completion；allocator peak 含此前用例，不作为
单阶段增量。冷启动单帧用例含编译，原始回执另存。这是单次正确性运行，
不是 SP scaling 基准；exact 对照限定同一 SP 的同一诊断服务，不要求跨 SP
或不同服务重启之间 RGB exact。

硬件按实际回执记录：NVIDIA L20D、CC 10.3、274,113 MiB/卡，
Torch 2.13.0+cu130 / CUDA 13.0。replicated encoder、SP=Ulysses、ring=1、
torch_sdpa、speed 模式、warmup off、TAEH3 FP16，与上一轮的采样配置相同。

### 证据索引

本轮 [结构化回执](source_presentation_review_acceptance_20260923.json) 包含每个会话、逐 rank 指标、129 项比较、
32 项边界结果、三路 native 参考报告及归档 SHA-256/大小。66 个权重/组件
重新逐文件哈希；清单 SHA-256：`a37bf8bf8fbe36caa344c2f43c7599ccc598b656801b774c298c501755cb46c5`。
模型配置 SHA-256：`e1b18056739898af4dd712c18775f11a3757a6c1cb744ae0b60e5a61698b9850`。

零 GPU 导出作业 `dlcf5a2a2o88kcr4` Succeeded。原始张量、RGB、请求、服务日志、
GPU 采样和检查脚本保存在私有 OSS：
`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing173-20260923/review-r2/export-r2/`。
`metadata.tar.gz` 已回读校验 360,637 bytes，SHA-256
`a213c8700d32860039ca5f7340b026853e4c79d8ed0b45f6f9627d31c738774a`。
对应 CPFS 根目录：`/cpfs/world-model/sglang/evals/zing173-20260923/review-r2`。
metadata 中保存 matrix、lifecycle、boundaries、observer 和本轮内存 processor
审计代码；完整 processor/SP1/SP2 归档另存，均有服务端 SHA-256/大小回执。
获取这些证据需要已有私有 OSS/CPFS 权限，没有发布公开媒体页。

此前 `7a7aa118` 的 [历史回执](source_preview_acceptance_20260923.json) 原样保留，
其开关/PNG 路径和元数据诊断不作为当前内存实现验收。#167 native learned
features/latent、音频、24 FPS 吞吐和画质继续独立跟踪，本轮不声称通过。
