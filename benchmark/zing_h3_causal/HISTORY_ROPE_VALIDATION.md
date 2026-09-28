# History K RoPE 验证（#179）

当前状态：实验性 opt-in，默认保持 eager。本文记录原逐片段 fused 的冻结实验：单测、真实 collective、确定性配置的长窗口数值与 120 个正式样本的端到端性能门槛已通过；独立 profile 和 14 份 Perfetto UI 核对已完成。原 performance encoder 的严格数值重复性门槛、实际张量尺寸微基准的耗时波动门槛未通过；source 跟随画质存在自动抽样标记的问题，最终视觉验收待人工判断。

根据 MinWM 对照继续实现的 grouped、连续视图和当前 forward metadata 复用，见[后续实现与验证记录](HISTORY_ROPE_GROUPED_PLAN.md)。这些路径独立 opt-in，重新冻结 main 外围基线验证；本文历史 120 个样本和收益不能用于证明新路径已通过端到端验收。PR 当前为非 draft，不自动合并。

## 视频入口与人工验收

[打开同步对比页](https://review-cdn.loopit.com.cn/reports/zing179-20260923/playback-r1/index.html)（办公网或公司 VPN）。每组点击“从头同步播放”，整组缓冲后一起起播；也可单独播放。七组均已实际在浏览器播放并检查出帧；MP4 是播放用转码，逐位一致结论来自原始 RGB 哈希，不来自有损 MP4。

| 要判断的问题 | 对照与入口 | 当前证据及判断边界 |
| --- | --- | --- |
| 原 performance encoder 是否可重复 | [两次独立 eager A₁/A₂，15 秒](https://review-cdn.loopit.com.cn/reports/zing179-20260923/playback-r1/index.html#repeat) | 这是 A/A 控制，两侧均关闭候选。首个 prefix forward 前已有输入张量差异，故严格数值重复性失败；肉眼相似不能代替该门槛，具体上游非确定性原因尚未定位 |
| 原 performance encoder 的 eager/fused 画面 | [SP1 A/B](https://review-cdn.loopit.com.cn/reports/zing179-20260923/playback-r1/index.html#old-sp1)、[SP2 A/B](https://review-cdn.loopit.com.cn/reports/zing179-20260923/playback-r1/index.html#old-sp2) | 保留原实验输出；由于该配置 A/A 已不重复，不能把 A/B 差异归因于本 PR。原 SP4 仅完成 baseline，不能构造缺失的候选视频 |
| 确定性配置是否跟随 source | [Source / eager / fused，四组 10/15 秒视频](https://review-cdn.loopit.com.cn/reports/zing179-20260923/playback-r1/index.html#quality) | 同 SP/时长 A/B 原始 RGB 完全相同，页面复用去重文件；source 跟随的审美与动作质量由人工最终判断。SP1/SP2 输出相同，合并展示 |
| 微基准的耗时是否平稳 | [20 次原始计时及 12 组门槛表](https://review-cdn.loopit.com.cn/reports/zing179-20260923/playback-r1/index.html#micro) | 算子微基准没有生成视频；这里的“稳定性”专指耗时波动，与画质、正确性分开 |

**画质状态：待人工验收。** 下文“停留甲板过久、动作不足、几何漂移”等是本轮全时段 4fps 抽样的观察及待复核时间点，不是用户或评审人的最终结论。历史 `quality-review.json` 中的 `NOT PASSED` 保留为自动审查记录，不改写原证据，也不将其当作人工拒收。未得到人工判断前不宣称画质通过；两侧相同则证明本次优化在该配置没有引入 RGB 差异。

source 原片为 5 秒，长请求重复两次/三次；5/10 秒是输入拼接点，不能把该边界直接认作候选引入的画质问题。重复 source 播放版只供对照，不是新生成结果。A/A 视频由零 GPU 导出作业 `dlcy82yoasebrkht` 从原运行的两份 RGB 生成，逐份核对原 RGB 与 summary 哈希，未重新推理；媒体来源及哈希见播放页 manifest 和 `aa-playback/manifest.json`。

## 实现与边界

`SGLANG_ZING_FUSED_HISTORY_ROPE=1` 启用融合历史 K 重建。默认值为 `0`。
融合路径仅支持无 autograd 的 CUDA BF16、连续 rotary 系数和连续输出；输入允许行间 stride。
不支持的 dtype/layout/device/alias 回退现有 `_apply_rope_into`。
保留每次乘法的 BF16 舍入再相加，原始 normalized K 不变，非 rotary 尾部直接复制原始位模式。
缓存失效策略、当前 KV writer、QK norm 和 FA4 metadata 均使用冻结基线逻辑。

## 与已有 MinWM 实现的对照

**MinWM 已有类似功能，分为两层：复用旋转后的 K，和融合一次 RoPE 的运算。** 2026-07-31 的 [a465859d：缓存与 metadata 复用](https://github.com/seedleap/sglang/commit/a465859d2ea0c7b75b81c4e15455e7f1d32d02a4) 引入 `MINWM_CACHE_ROTATED_K` 与跨层计划复用；[58c4de38](https://github.com/seedleap/sglang/commit/58c4de3887b6ca5c3c74e80b5f06e0e8017e3428) 继续减少缓存与 Ulysses 布局开销。2026-08-25 的 [#64 / 9e0238a0](https://github.com/seedleap/sglang/commit/9e0238a042ef672e9177aba30072cc96c668ccf1) 实现 Hopper 的单 kernel rotary。下表对照本 PR 冻结基线 `e6537184` 中的 MinWM 代码，不代表其他分支或历史 Blackwell 扩展的全部状态。

| 维度 | MinWM（冻结基线） | 本次 Zing #179 |
| --- | --- | --- |
| 缓存语义 | 缓存 RMSNorm 后、RoPE 前的 raw K；选 sink/pin/tail 窗口，再按可见顺序构造 block-relative 位置 | 同样保留 normalized raw K；按 Zing 的 prefix、source/target 片段与有效位置重建历史 K |
| 避免重复旋转 | `MINWM_CACHE_ROTATED_K` 默认开；有效缓存且 `is_recompute` 时仅重算当前 K，否则旋转整个可见 K；非 recompute 更新使 rotated K 失效 | Zing 基线已按 raw K/有效位置签名复用已旋转历史的共同前缀；#179 不新增或修改这项失效策略，只加速必须重建的历史片段 |
| 运算融合的范围 | `apply_minwm_rotary_embedding[_out]` 用于 Q 和 K；适配时单个 CUDA kernel 写调用方输出，fallback 为 eager 后 copy | `apply_history_rope_into` 仅历史 K；每个失效片段一次 Triton launch，直接写目标 K；当前 KV writer 与 Q 路径保持基线 |
| 布局与算术 | interleaved 偶/奇配对，全 D128；FP32 cos/sin、FP32 乘加，最后转换 BF16/FP16；CUDA 显式乘法与加减，保持 eager 舍入 | rotate-half 配对，实际 D128/R96，尾部 32 维原样保留；BF16 系数，每次乘法先舍入 BF16 再相加，禁止 FMA 合并 |
| 设备与布局约束 | 本快照 fast path 要求 SM90、BF16/FP16、连续输入/输出；不满足则走模型 wrapper fallback | CUDA 非 HIP、无 autograd、BF16；允许输入 row stride，连续系数/输出，拒绝 alias；默认 opt-in 关闭 |
| 跨层系数复用 | attention plan 的 query/key cos/sin 可预计算并供各层复用 | 既有 cache 按 position key/dtype 复用系数；本 PR 不改变其策略 |

源码定位：[MinWM attention 分派](../../python/sglang/multimodal_gen/runtime/models/dits/minwm.py)（`apply_minwm_rotary_embedding_out`、`MinWMCausalSelfAttention.forward`）、[MinWM 缓存](../../python/sglang/multimodal_gen/runtime/models/dits/minwm_kv_cache.py)（`_apply_attention_plan`）、[MinWM fast-path 条件](../../python/sglang/kernels/ops/diffusion/rope/minwm_rotary_jit.py)、[MinWM CUDA 算术](../../python/sglang/kernels/jit/csrc/diffusion/minwm_rotary.cuh)；对照 [Zing 缓存复用](../../python/sglang/multimodal_gen/runtime/models/dits/zing_h3_cache.py)（`raw_attention_view`）和 [本次 K-only kernel](../../python/sglang/kernels/ops/diffusion/rope/zing_history_rope.py)。已有 MinWM 测试 `test_minwm_five_passes_rotate_q_five_times_and_full_k_once` 明确检查五次 forward 的 Q 长度为 `[2,2,2,2,2]`、K 为 `[4,2,2,2,2]`；这是已有代码证据，本次文档修订未声称重跑 MinWM GPU 性能。

可复用的是“保留 raw K、缓存派生的 rotated K、只重算失效部分、直接写预分配输出”的设计。不能直接拿 MinWM kernel 替换 Zing：配对方式、系数 dtype、中间舍入、partial RoPE 和输入 stride 都不同，会破坏当前严格逐位语义。MinWM 全可见窗口一次 launch 与本节冻结 Zing 逐失效片段 launch 的粒度也不同；下文历史 pack/grouped 数据属于当时未接入模型的原型。后续 runtime opt-in 的新增实现与新基线验证单列在[后续记录](HISTORY_ROPE_GROUPED_PLAN.md)，没有用代码对照推断两模型的相对端到端性能。

## 冻结配置

- 基线：`e6537184cdcd9de05e04da6201be70446847b5be`。
- 运行时代码：`c82cce1327e1502f031c7691514362f117f112ea`；后续提交修正测试夹具并整理复现脚本，不改变 kernel/dispatcher。
- 镜像：`zing-thailand-registry-vpc.ap-southeast-7.cr.aliyuncs.com/zing/sglang@sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`。
- PAI：`ap-southeast-7`，`shengdong`，workspace 384；产物根 `oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing179-20260923/`。
- 实测设备报告 NVIDIA L20D，CC 10.3，274113 MiB/card。单卡微基准与四卡端到端分别分配；四卡节点两两 NV18。
- PyTorch 2.13.0+cu130、CUDA 13.0、Triton 3.7.1、FA4 4.0.0b19、NCCL 的 PyTorch API 报告值 2.29.7；实际 collective 服务日志另打印 2.30.7+cuda13.3。二者来自同一冻结镜像，未据此推断差异原因。

端到端使用真实 `zing-whitebox-video-five-triview-0001` source + 六张参考图，832×480、24fps、step1000 权重、四步 `[1000,750,500,250]`、guidance 1、seed 42、`indexed-cpu-v1`，progressive H3 encoder 与 TAEH3 fp16。window21/sink3/block_relative/gap12，无 offload。原素材为 5 秒视频，10/15 秒输入通过重复两次/三次构造，存在拼接边界，不是独立长素材。当前入口仅验收视频。

预先门槛：history rebuild 中位数改善 >10%，A→B 与 B→A 的端到端中位数均改善 >3%，各配置 `(max-min)/median <=10%`。每配置每顺序五个正式样本；先独立数值诊断，性能关闭 profiler 和 tensor clone/hash，独立采集 profile。数值门槛为同 SP 逐位一致，不在测量后放宽。

## 已完成检查

| 作业 | 结果 |
| --- | --- |
| `dlcb1r11mpo3yd05`（CPU） | 151 passed，54 CUDA skips；完整模型哈希 |
| `dlc1l3vmkqaj1g56`（1 GPU） | 106 passed；微基准完成 |
| `dlc13wm7ey4wi770`（4 GPU，首轮） | 106 passed；collective 测试替身 `Identity` 缺少 QK norm 权重字段，正式 A/B 前退出；已改真实 RMSNorm |
| `dlc1g49yiywel2lg`（4 GPU，重跑） | 106 passed；SP1/2/4 所有 rank 的 collective 输出、捕获 KV 和原始 K 保留检查通过，融合路由命中；SP1/2 A/B 和 SP4 baseline 各完成 360 帧；SP4 baseline 后 stdout 拼行触发结果解析失败，未执行 SP4 candidate 和正式性能 |
| `dlc114ro5mxsi6rp`（CPU） | 导出微基准，补齐 text encoder/audio 等额外资产哈希 |

首轮 CPU 准备也保留了 pytest 导入环、prefix 夹具帧数和传输末尾换行导致校验失败的记录；这些失败没有产生有效性能样本。

## 长窗口诊断与未通过项

零 GPU 作业 `dlc1mybyi4y1ze97` 从原始日志离线恢复了上述五组结果，未改动原实验。
每组每个 rank 均有 65 个唯一且连续的生成块、43 次实际淘汰，360 帧全部提交、接收、消费和返回，最终缓存清理检查通过。
解析器修订使用 JSON decoder 逐条读取同一行中的多 rank 记录；畸形记录仍报错，不静默丢弃。

原 `vae_config.encode_mode=performance` 配置的端到端逐位门槛**未通过**：

| 比较 | source latent 不同的块数 | 最终 latent 不同的块数 | noise / effective positions |
| --- | ---: | ---: | --- |
| SP1，rank0 | 43/65 | 65/65 | 全部逐位一致 |
| SP2，rank0 | 65/65 | 65/65 | 全部逐位一致 |
| SP2，rank1 | 65/65 | 65/65 | 全部逐位一致 |

SP1 candidate，以及 SP2 candidate 的每个 rank，各 118750 次真实 history rebuild、219680500 行均通过相对于 eager 的逐位检查，raw K 保持不变，融合路由全部命中。
这是每次 history rotation 的正确性证据，不能替代包含上游输入差异的全模型 A/B 验收。
SP1 block0 的 source/noise 相同，但后续 latent 和 post-step KV 快照不同；post-step prefix 快照不能用于推断差异最初出现的位置。
独立进程 baseline A/A 作业 `dlc1lenbbethfgps` 已完整运行两次 360 帧；两次均禁用候选优化，数值门槛仍未通过。
forward 前的 prefix 输入 `x [5040,4,24]` 已有 30549/483840 个元素不同（max abs 0.03515625），context、时间步、位置、角色和长度一致。
65 个块的 source 与最终 latent 全部不同，noise 和 effective positions 全部相同。
首次 QKV 投影、normalized QK、首次 KV store 和 prefix 输出也不同；因此不能把已观察到的输入差异归因于本次 RoPE kernel。
这些证据没有定位到具体 cuDNN 算法，未作此推断。

五组视频各 360 帧均完整解码通过；全时段每秒四帧抽样发现 baseline 与 candidate 均对 source 的场景与运动跟随不足，部分时段（包括约 10–11.5 秒）出现重影和几何漂移。
这属于 baseline 已有质量问题，不归因于候选优化；原素材在 5/10 秒的重复拼接另行保留为限制。
原 performance 配置这轮没有正式无 profiler 的端到端性能样本，不以诊断耗时推算性能收益。
原始恢复记录位于产物根的 `full-r2-recovered/`，A/A 记录位于 `aa-r1-audit/`。

## 单独冻结的确定性实验

`dlc1yq81e2cjdnhe`（零 GPU）准备独立 `model-deterministic-v1` 目录，保留原模型不变。
仅模型配置中的 `vae_config.encode_mode` 从 `performance` 改为 `deterministic`，并更新 profile 的配置校验哈希；BF16 encoder dtype、posterior seed 和全部权重不变。
新实验另设 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，两侧相同；idle/lifetime 为 3600/7200 秒。

- config SHA256：`325c3d20f7149a040c7bd36e36749af4d2d0c1125379483652635cd28952cb93`。
- profile SHA256：`aafa803caa54d691c5924f22efd9aa262d0a5fbb3f69ae63d8702ec78e98f7f6`。
- 配置 JSON 语义、镜像与 #158 已通过的 15 秒 baseline A/A 相同；65 个共同模型资产哈希一致，TAEH3 路径一致。不同 JSON 序列化产生各自配置文件哈希，不混用文件。
- 本任务独立矩阵 `dlcyds1e2kitlpr0` 已成功完成。每个 SP 的 A/B 必须通过 source/noise/cache/positions/final latent、prefix entry/首次写入及最终 RGB 的严格比较，才继续正式性能阶段。

SP1 的 15 秒 A/B 已通过严格门槛：65 个块及六份 prefix 快照逐位一致，最终 RGB SHA256 为 `1160535e35e15dd598a0fcf0aca920888334407592e25046acbdc9b8177edef3`，与共享 baseline 输出一致。零 GPU 作业 `dlc1xcpiq2vajk7y` 分别核验两份 RGB，并完成全部 360 帧解码。全时段 4fps 抽样仍发现人物在甲板停留过久、3–8 秒动作不足以及约 9–12.5 秒场景几何漂移；source 跟随质量被自动抽样标记为待复核，最终待人工验收。两侧 RGB 相同，故此配置没有候选引入的画面差异，但不能据此宣布画质验收通过。SP2 的两个 rank 随后也各通过 65 块及六份 prefix 快照严格比较，最终 RGB 与上述 SHA256 相同；SP4 的四个 rank 也均通过相同严格门槛，SP4 两侧 RGB SHA256 为 `dbbac3fa3698b34d73b441bd7d9a44c33e15e2792c5761fe534280c2a2834abe`。不要求跨 SP 输出相同。全部正式样本及其输出核验已经完成，结果见下节。
零 GPU 作业 `dlcm6qrd63mm2z8e` 随后核验全部六份 RGB 并完整解码两种唯一输出。SP4 全时段自动抽样也标记了以下 source 跟随问题，待人工复核：多数时间仍在甲板/楼梯，约 2–6 秒动作不足，6–10 秒栏杆与地面几何漂移，约 6.25/11.25 秒有人物模糊或双影。SP4 两侧 RGB 相同，未发现候选引入的视觉差异。此抽样不等同逐帧运动审查。
产物位于 `full-r3-monitor-r18/diagnostic-sp{1,2,4}-comparison.json` 与 `full-r3-diagnostics-media/`。

共享 A/A 仅证明该确定性配置的基线可复现，不代表本任务候选正确性、性能或画质已通过；新结果也不覆盖原 production performance 配置。

## 正式输出的完整覆盖与画质限制

零 GPU 作业 `dlcp0b3t6djjvrs9` 已逐一校验全部 120 份正式 RGB，四种唯一输出分别完整解码 240 或 360 帧。15 秒输出与上述已审查诊断输出 SHA256 相同，复用其哈希关联的视觉审查；另外两种 10 秒输出已查看全时段 4fps 联系表。

- SP1/SP2 的 10 秒输出 SHA256：`e7317f7f7677f0b795379d19f209239cd6b3f225f855e12488b0c8daf12ea0f5`。人物全程仍在甲板，约 3–8 秒近乎静止，栏杆、镜头与布局变化，约 8.5 秒后恢复动作但仍未按 source 离开甲板。
- SP4 的 10 秒输出 SHA256：`6628f2b0435923d936271502ad5eb14d6ce942409d26cd2b30619d351a6aed2b`。人物全程在甲板/楼梯附近，约 2–6 秒动作不足，6–10 秒跑动伴随栏杆消退与楼梯几何漂移，约 6.25 秒抽样帧出现模糊或双影。
- 四种输出均有自动抽样标记的 source 跟随问题，最终质量结论待人工验收。铅笔纹理、发髻、背心与服装颜色大体对应，完整六参考图忠实度未建立。同 SP/时长所有 A/B 输出完全一致，因此未发现候选引入的视觉差异。

审查包括全时间范围抽样和逐帧技术统计，不等同逐帧运动审查。原始 source 只有 5 秒，10/15 秒请求分别重复两次/三次，5 秒与 10 秒拼接点已披露。`full-r3-final/quality-review.json` 将四个唯一输出的审查关联到全部 120 个样本。994 份原始数值 `.pt`（15.687 GiB）和四份去重 RGB 已保存在同目录的 OSS `numerical/`、`raw-rgb/`，源文件与目标挂载回读 SHA256 均核验，映射见 `retention-manifest.json`。

## 正式端到端性能（确定性配置）

`dlcyds1e2kitlpr0` 完成全部 120 个无 profiler 正式样本；每个进程先完整预热 15 秒和 10 秒。零 GPU 作业 `dlcp0b3t6djjvrs9` 对原始 RGB 文件逐个重新计算 SHA256，并检查每个 rank 的路由、连续块编号和至少三次实际淘汰。

所有 12 项预设端到端门槛通过，最大组内 spread 为 7.1446%。这只适用于单独冻结的确定性配置，不覆盖原 performance encoder 配置，也不代表画质通过。独立满窗口 profile 已完成；实际张量尺寸微基准中位数改善，但多数组合的耗时波动门槛未通过，见后文。

| SP | 顺序 | 输入秒数 | Eager 中位数 s | Fused 中位数 s | 降幅 | Eager spread | Fused spread | 门槛 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | AB | 10 | 46.404 | 38.095 | 17.91% | 4.62% | 3.06% | 通过 |
| 1 | AB | 15 | 69.336 | 57.318 | 17.33% | 1.63% | 2.39% | 通过 |
| 1 | BA | 10 | 46.261 | 39.447 | 14.73% | 7.14% | 1.33% | 通过 |
| 1 | BA | 15 | 69.678 | 58.296 | 16.33% | 2.64% | 4.39% | 通过 |
| 2 | AB | 10 | 45.369 | 42.706 | 5.87% | 3.97% | 3.97% | 通过 |
| 2 | AB | 15 | 68.086 | 63.484 | 6.76% | 1.68% | 0.93% | 通过 |
| 2 | BA | 10 | 45.136 | 42.998 | 4.74% | 3.43% | 1.61% | 通过 |
| 2 | BA | 15 | 67.206 | 63.225 | 5.92% | 3.61% | 4.05% | 通过 |
| 4 | AB | 10 | 48.257 | 46.244 | 4.17% | 3.81% | 4.38% | 通过 |
| 4 | AB | 15 | 72.745 | 69.526 | 4.42% | 2.84% | 2.92% | 通过 |
| 4 | BA | 10 | 48.843 | 46.508 | 4.78% | 4.07% | 4.57% | 通过 |
| 4 | BA | 15 | 73.357 | 69.336 | 5.48% | 1.19% | 4.75% | 通过 |

### 全部正式 API wall 样本

单位秒；每个单元格按运行顺序列出五次测量。预热不计入这些样本。

| SP | 顺序 | 输入秒数 | Eager 五次 | Fused 五次 |
| ---: | --- | ---: | --- | --- |
| 1 | AB | 10 | 45.963, 46.404, 46.904, 46.193, 48.109 | 38.750, 37.673, 38.095, 38.302, 37.585 |
| 1 | AB | 15 | 69.232, 69.757, 70.092, 69.336, 68.961 | 56.582, 57.318, 57.953, 56.794, 57.791 |
| 1 | BA | 10 | 45.979, 46.261, 46.488, 45.652, 48.957 | 39.447, 39.686, 39.649, 39.193, 39.161 |
| 1 | BA | 15 | 70.144, 70.485, 69.678, 68.647, 68.742 | 57.317, 59.876, 58.070, 58.347, 58.296 |
| 2 | AB | 10 | 45.093, 45.087, 46.890, 46.124, 45.369 | 42.706, 42.357, 44.054, 43.597, 42.455 |
| 2 | AB | 15 | 68.124, 68.649, 67.502, 68.070, 68.086 | 63.244, 63.714, 63.241, 63.484, 63.829 |
| 2 | BA | 10 | 44.725, 44.475, 45.291, 45.136, 46.025 | 42.403, 43.044, 43.033, 42.351, 42.998 |
| 2 | BA | 15 | 67.206, 67.933, 67.170, 66.276, 68.705 | 63.522, 63.075, 63.225, 63.222, 65.638 |
| 4 | AB | 10 | 47.383, 47.379, 49.217, 48.394, 48.257 | 46.215, 45.795, 47.257, 46.244, 47.819 |
| 4 | AB | 15 | 72.567, 73.353, 71.415, 72.745, 73.482 | 68.845, 70.595, 68.568, 69.526, 70.127 |
| 4 | BA | 10 | 48.843, 47.959, 49.947, 48.845, 48.810 | 45.884, 46.119, 46.508, 48.008, 47.943 |
| 4 | BA | 15 | 73.675, 73.502, 72.813, 72.801, 73.357 | 68.342, 69.430, 68.562, 69.336, 71.633 |

### 逐 rank 主机阶段耗时

表中 DiT 为每个请求的累计 `dit_host_ms` 中位数（秒），满窗为起始时至少保留 21 个 chunk 的生成块主机耗时中位数（毫秒），包括满足条件的尾块/EOS。二者均不是 CUDA kernel 时间；不累计 rank。

| SP/rank | 顺序 | 输入秒数 | DiT eager/fused s | 满窗 eager/fused ms |
| --- | --- | ---: | ---: | ---: |
| 1/0 | AB | 10 | 33.732 / 25.781 | 876.272 / 629.458 |
| 1/0 | AB | 15 | 51.521 / 39.441 | 875.731 / 640.036 |
| 1/0 | BA | 10 | 33.877 / 26.906 | 879.044 / 653.181 |
| 1/0 | BA | 15 | 51.528 / 40.143 | 873.766 / 644.235 |
| 2/0 | AB | 10 | 32.924 / 30.222 | 804.763 / 712.553 |
| 2/0 | AB | 15 | 50.139 / 45.609 | 781.885 / 706.317 |
| 2/0 | BA | 10 | 32.620 / 30.403 | 778.634 / 708.974 |
| 2/0 | BA | 15 | 49.353 / 45.261 | 779.803 / 711.976 |
| 2/1 | AB | 10 | 32.863 / 30.192 | 804.587 / 709.916 |
| 2/1 | AB | 15 | 50.007 / 45.558 | 781.510 / 704.836 |
| 2/1 | BA | 10 | 32.556 / 30.018 | 777.185 / 707.801 |
| 2/1 | BA | 15 | 49.236 / 45.237 | 776.667 / 711.213 |
| 4/0 | AB | 10 | 35.488 / 33.542 | 856.064 / 798.284 |
| 4/0 | AB | 15 | 54.333 / 51.375 | 862.363 / 796.063 |
| 4/0 | BA | 10 | 36.045 / 33.295 | 875.837 / 788.041 |
| 4/0 | BA | 15 | 55.011 / 50.838 | 868.383 / 790.638 |
| 4/1 | AB | 10 | 35.371 / 33.401 | 851.672 / 793.745 |
| 4/1 | AB | 15 | 53.682 / 51.204 | 861.742 / 793.907 |
| 4/1 | BA | 10 | 35.956 / 33.226 | 872.676 / 788.799 |
| 4/1 | BA | 15 | 54.755 / 50.776 | 866.553 / 790.144 |
| 4/2 | AB | 10 | 35.332 / 33.477 | 852.505 / 796.335 |
| 4/2 | AB | 15 | 53.953 / 51.345 | 861.271 / 795.172 |
| 4/2 | BA | 10 | 36.049 / 33.134 | 873.802 / 788.015 |
| 4/2 | BA | 15 | 54.992 / 50.556 | 868.975 / 787.770 |
| 4/3 | AB | 10 | 35.394 / 33.457 | 852.196 / 796.655 |
| 4/3 | AB | 15 | 53.828 / 51.146 | 860.731 / 794.756 |
| 4/3 | BA | 10 | 36.033 / 33.182 | 873.891 / 788.882 |
| 4/3 | BA | 15 | 54.442 / 50.704 | 866.330 / 789.700 |

### 逐 rank 显存与 GPU 秒

以下峰值是独立服务进程的高水位，包含预热；每个 SP/rank/variant 对两种顺序和两种时长取最大值，单位 GiB。不是稳态活跃张量大小。

| SP/rank | Eager allocated/reserved | Fused allocated/reserved |
| --- | ---: | ---: |
| 1/0 | 229.413 / 229.941 | 229.413 / 229.941 |
| 2/0 | 182.341 / 182.832 | 182.341 / 182.832 |
| 2/1 | 182.379 / 182.949 | 182.379 / 182.949 |
| 4/0 | 158.782 / 159.414 | 158.782 / 159.414 |
| 4/1 | 158.819 / 159.434 | 158.819 / 159.434 |
| 4/2 | 158.819 / 159.336 | 158.819 / 159.336 |
| 4/3 | 158.819 / 159.336 | 158.819 / 159.336 |

每个样本分别记录参与 GPU 秒 `SP × API wall` 与四卡分配请求秒 `4 × API wall`。它们是请求窗口的资源记账，不是实测 GPU busy time，也不包含进程加载、预热和诊断。完整数值见产物 `full-r3-final/analysis.json`；逐样本与逐 rank CSV 同步保留在本地报告归档。

### 正式输出一致性

每个 SP/时长均有 20 次正式输出（两种顺序 × 两种实现 × 五次）。同组 input/output SHA256 全部相同；15 秒输出也与对应严格诊断精确匹配。120 个原始 RGB 全部核验并完整解码四种唯一输出。

| SP | 输入秒数 | 正式 RGB SHA256 |
| --- | ---: | --- |
| 1 | 10 | `e7317f7f7677f0b795379d19f209239cd6b3f225f855e12488b0c8daf12ea0f5` |
| 1 | 15 | `1160535e35e15dd598a0fcf0aca920888334407592e25046acbdc9b8177edef3` |
| 2 | 10 | `e7317f7f7677f0b795379d19f209239cd6b3f225f855e12488b0c8daf12ea0f5` |
| 2 | 15 | `1160535e35e15dd598a0fcf0aca920888334407592e25046acbdc9b8177edef3` |
| 4 | 10 | `6628f2b0435923d936271502ad5eb14d6ce942409d26cd2b30619d351a6aed2b` |
| 4 | 15 | `dbbac3fa3698b34d73b441bd7d9a44c33e15e2792c5761fe534280c2a2834abe` |

## 微基准

22 chunks（5040 + 21×780 行）、D128/R96，每种配置预热 20 次、测量 20 次，单独采集 trace。
下表为 CUDA event 区间中位数（包含主机提交间隙），单位 ms；不是纯 kernel 时间总和。

| Heads | Eager | 当前逐块融合 | 拼接后融合 | 单 launch 分组原型 |
| --- | ---: | ---: | ---: | ---: |
| 14 | 1.810 | 0.678 | 0.140 | 0.115 |
| 28 | 1.801 | 0.647 | 0.206 | 0.172 |
| 56 | 2.067 | 0.642 | 0.348 | 0.277 |

对应 kernel 数为 176 / 22 / 3 / 1。拼接方案在计时内执行 `torch.cat`；分组原型在计时内构造 offsets/launch metadata。后两者仅用于评估，没有接入模型路径。分组原型尚未覆盖真实动态缓存布局、编译成本和端到端行为，不能作为已验收生产优化。原始 36 组记录和各组全部样本位于 `export-kernel-r1/micro.json`。`logical_bytes` 仅为唯一输入、输出与系数张量字节量，不代表实测 DRAM 流量。

复现单卡微基准（使用上述镜像、隔离 GPU，并将本 checkout 加入 PYTHONPATH）：

```bash
OUT=/tmp/history-rope-benchmark \
SGLANG_LIGHTWEIGHT_RUNTIME=1 \
SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cuda \
python benchmark/zing_h3_causal/benchmark_history_rope.py
```

复现真实 FA4/Ulysses collective：

```bash
for sp in 1 2 4; do
  torchrun --standalone --nproc-per-node="$sp" \
    benchmark/zing_h3_causal/validate_history_rope_collective.py
done
```

## 实际历史窗口尺寸的微基准

独立作业 `dlckuh72ih9wedh1` 在另一节点 `lingj125bqx6g0v0-mg6chjbeabglj0jv-0` 完成 12 种尺寸/阶段配置 × 4 条路径的 48 条测量记录；实测同为 L20D、CC 10.3、274113 MiB/card，镜像与运行时代码不变。张量尺寸从原诊断的 block30/31 还原，source 为 21 个片段，target 为 22 个片段（含 pending source），总行数 37496–38666，D128/R96。输入是随机 BF16，**不是原 cache 数值回放**。四种路径均逐位匹配 eager；每组 20 次预热、20 个完整计时样本。

旧文所称“真实几何”，指从真实运行还原片段长度、head 数与 D/R 等张量尺寸；每个输入 chunk 独立分配为连续张量，并未复现真实 cache 的共享 storage，也不表示真实画面生成或真实 cache 数值回放。12 组 = 2 个 block（30/31）× 2 个阶段（source/target）× 3 种 head 数（14/28/56）；每组测 eager/fused/pack/grouped 四条路径，因此原始记录共 48 条。稳定性统计只对 12 对 eager/fused 分别检查两侧的 20 次 CUDA-event 区间：`spread = (max(samples) - min(samples)) / median(samples)`，两侧都 ≤10% 才算该对通过。

例如 H56 block31-target 的 fused：min **0.624032 ms**、median **0.669408 ms**、max **1.713760 ms**，spread **162.79%**。这是“个别调用明显慢于典型调用”的证据，不是数值错误、视频闪烁或整体比 eager 更慢的证据；该组 eager 中位数为 3.1591 ms。11 对的 fused 超过 10%，其中 1 对的 eager 也超过；唯一两侧通过的是 H14 block31-source。CUDA-event 区间会包含 GPU 等待 host 继续 enqueue 的空隙，本记录不能区分 GPU 执行、host 调度或其他原因，尚未完成波动根因归因。不能仅凭长尾把问题归到 Triton kernel。

以下为 CUDA-event 区间中位数（ms），包含 CPU enqueue 空隙；pack 包含 concat，grouped 包含元数据准备。pack/grouped 仅为微基准原型，没有接入模型。独立 profiler 的 kernel 累计成本另存 CSV，不能与此区间或端到端 wall time 混用。

| Heads | Geometry | Rows | Eager | Fused | Pack | Grouped | Reduction | Eager spread | Fused spread |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | block30-source | 38276 | 1.3565 | 0.6324 | 0.1972 | 0.1535 | 53.38% | 10.06% | 21.35% |
| 14 | block30-target | 38666 | 2.0364 | 0.6562 | 0.2033 | 0.1582 | 67.78% | 3.34% | 13.59% |
| 14 | block31-source | 37496 | 1.9435 | 0.6198 | 0.1937 | 0.1532 | 68.11% | 4.08% | 9.83% |
| 14 | block31-target | 38276 | 2.0324 | 0.6464 | 0.2010 | 0.1558 | 68.19% | 3.69% | 12.08% |
| 28 | block30-source | 38276 | 1.9491 | 0.6279 | 0.3238 | 0.2499 | 67.79% | 3.89% | 14.95% |
| 28 | block30-target | 38666 | 2.0266 | 0.6624 | 0.3305 | 0.2534 | 67.32% | 3.68% | 12.22% |
| 28 | block31-source | 37496 | 1.9404 | 0.6379 | 0.3189 | 0.2442 | 67.13% | 4.27% | 12.62% |
| 28 | block31-target | 38276 | 2.0352 | 0.6702 | 0.3273 | 0.2519 | 67.07% | 4.44% | 12.07% |
| 56 | block30-source | 38276 | 3.1468 | 0.6476 | 0.6056 | 0.4276 | 79.42% | 0.66% | 17.71% |
| 56 | block30-target | 38666 | 3.1863 | 0.6660 | 0.6128 | 0.4327 | 79.10% | 0.98% | 15.71% |
| 56 | block31-source | 37496 | 3.0898 | 0.6315 | 0.5933 | 0.4185 | 79.56% | 0.93% | 11.29% |
| 56 | block31-target | 38276 | 3.1591 | 0.6694 | 0.6082 | 0.4310 | 78.81% | 0.92% | 162.79% |

12 组 fused 中位数均改善 >10%（53.38%–79.56%），但 **11/12 对未满足组内 spread ≤10%**，不能声明微基准稳定性门槛全通过。尤其 H56 block31-target 的 fused spread 为 162.79%；所有长尾样本保留，不剔除、不事后放宽门槛。此波动结论与此前完整 120 样本端到端矩阵的 12 项门槛通过分别报告。

全部原始样本见 `profile-r1-final/micro.json`；同步 host 中位数/spread、CUDA-event ns/row、独立 kernel count/sum/ns per row 与 logical bytes 见 `micro-summary.csv`。Logical bytes 仅计唯一 K 输入/输出和系数，并非实测 DRAM 流量，也不覆盖 pack 的全部额外搬运。

同一次独立微基准 profiler 中，21/22 片段的 kernel 数从 eager 的 168/176 降到 fused 的 21/22；pack 为 3，grouped 为 1。下表每行成本是独立 profiler 的 kernel 累计时间除以总行数，不是 CUDA-event 区间或端到端延迟；范围覆盖上述四组片段尺寸。

| Heads | Eager ns/row | Fused ns/row | Pack ns/row | Grouped ns/row |
| ---: | ---: | ---: | ---: | ---: |
| 14 | 26.65–27.16 | 3.94–3.99 | 4.84–4.91 | 2.68–2.69 |
| 28 | 43.03–43.40 | 6.80–6.88 | 8.30–8.33 | 5.29–5.33 |
| 56 | 76.29–76.56 | 12.49–12.51 | 15.85–15.90 | 10.30–10.36 |

## 独立满窗口 profile

GPU 作业 `dlckuh72ih9wedh1` 与零 GPU 分析/导出作业 `dlc8nhbm2q7puior` 均成功完成。每个 SP/variant 使用独立进程，完整 15 秒预热后采集第二个 15 秒请求中的满窗口阶段；不与正式计时混用。14 份 trace 全部解析成功，每个 rank 均覆盖 3 个生成 step、3 次实际淘汰、18 次 forward（3 source、12 active、3 final）。A/B 每个 rank 的 history 重建均为 6450 次、11424300 行；baseline 路由为 disabled=6450，candidate 为 fused=6450，无候选回退。

### History 重建成本

逐调用 CPU scope 中位数下降 72.52%–75.42%，逐调用 CUDA kernel sum 中位数下降 83.40%–85.40%，均超过预设 >10% 中位数改善门槛。以下保留每个 rank，未跨 rank 求和。Kernel 总数均从 51600 降至 6450。每行成本是该 scope 的 CUDA kernel 总累计时间除以重建总行数，不是物理带宽或请求 wall time。

| SP/rank | CPU median ms A/B | CUDA sum median ms A/B | History CUDA total ms A/B | CUDA ns/row A/B |
| --- | ---: | ---: | ---: | ---: |
| 1/0 | 0.129552 / 0.033628 | 0.118721 / 0.019712 | 874.946 / 142.919 | 76.586 / 12.510 |
| 2/0 | 0.125685 / 0.034537 | 0.067521 / 0.010848 | 495.603 / 78.119 | 43.381 / 6.838 |
| 2/1 | 0.126618 / 0.033859 | 0.067361 / 0.010848 | 494.790 / 78.170 | 43.310 / 6.842 |
| 4/0 | 0.133208 / 0.032748 | 0.043618 / 0.006369 | 308.328 / 45.377 | 26.989 / 3.972 |
| 4/1 | 0.127043 / 0.033535 | 0.043390 / 0.006400 | 306.544 / 45.462 | 26.833 / 3.979 |
| 4/2 | 0.122740 / 0.032859 | 0.043552 / 0.006368 | 307.844 / 45.358 | 26.946 / 3.970 |
| 4/3 | 0.127254 / 0.032790 | 0.043329 / 0.006368 | 306.394 / 45.321 | 26.820 / 3.967 |

逻辑读写字节量 A/B 相同：SP1 每 rank 329750995200 B，SP2 每 rank 165972230400 B，SP4 每 rank 84082848000 B。它计入逐调用的 K 输入/输出与 rotary 系数，重复重建会重复计数；不是实际 DRAM 流量。

### 其他 scope 与通信

所有 rank 两侧均有 current writer 900 次、coefficients 129 次、reserve/grow 1200 次和 eviction compact 3 次。Writer 在 SP1 为 900 个 kernel，在 SP2/SP4 为 8100 个 kernel，两侧相同；coefficients 为 1032 个 kernel，两侧相同。下表各项为每个 rank 的累计值，单元格均是 eager / fused（ms）。未把当前 writer 的原有实现算作本 PR 新增收益。

| SP/rank | Writer CPU | Writer CUDA | Coefficients CPU | Coefficients CUDA | Reserve CPU | Compact CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/0 | 110.908 / 89.875 | 8.971 / 8.950 | 19.259 / 19.526 | 2.539 / 2.454 | 9.241 / 8.849 | 154.658 / 152.878 |
| 2/0 | 172.995 / 195.519 | 40.752 / 40.894 | 19.572 / 19.427 | 2.490 / 2.451 | 9.549 / 9.882 | 149.191 / 155.067 |
| 2/1 | 175.763 / 191.422 | 40.428 / 40.493 | 19.626 / 19.259 | 2.454 / 2.404 | 9.948 / 9.488 | 148.558 / 154.289 |
| 4/0 | 180.114 / 185.420 | 30.380 / 30.485 | 19.609 / 18.796 | 2.524 / 2.456 | 9.984 / 9.397 | 171.435 / 169.751 |
| 4/1 | 172.309 / 192.618 | 30.066 / 30.193 | 19.545 / 19.480 | 2.479 / 2.410 | 9.490 / 9.554 | 166.574 / 164.862 |
| 4/2 | 166.358 / 182.084 | 30.372 / 30.486 | 21.388 / 18.816 | 2.514 / 2.453 | 9.430 / 9.122 | 160.274 / 163.327 |
| 4/3 | 170.917 / 181.086 | 29.978 / 30.103 | 18.623 / 18.523 | 2.470 / 2.409 | 9.562 / 9.154 | 161.036 / 161.993 |

Reserve/grow 与 compact 没有关联到 CUDA **kernel**，不能据此认定没有设备操作：本统计不计 GPU memcpy/memset。Reserve scope 记录调用，不证明实际新分配字节数。完整 70 行 scope 数据见 `profile-scopes.csv`。

| SP/rank | Collective CUDA | Collective CPU | Host synchronization | Capture seconds A/B |
| --- | ---: | ---: | ---: | ---: |
| 1/0 | 0.000 / 0.000 | 0.000 / 0.000 | 55.090 / 58.058 | 7.179 / 5.054 |
| 2/0 | 345.316 / 107.191 | 666.147 / 678.979 | 102.922 / 61.400 | 8.755 / 7.044 |
| 2/1 | 333.891 / 565.176 | 656.315 / 655.876 | 98.631 / 129.253 | 8.726 / 7.019 |
| 4/0 | 523.944 / 1128.447 | 1138.192 / 1091.372 | 81.384 / 138.981 | 9.907 / 7.509 |
| 4/1 | 2523.612 / 860.740 | 1094.653 / 1098.360 | 284.832 / 87.723 | 10.034 / 7.492 |
| 4/2 | 2737.560 / 2160.891 | 1078.714 / 1067.723 | 355.171 / 197.873 | 9.901 / 7.534 |
| 4/3 | 2186.408 / 2264.171 | 1112.892 / 1056.141 | 217.263 / 231.544 | 9.914 / 7.473 |

NCCL kernel 时间包含传输与设备端等待，无法从这些 trace 单独拆分。CPU wait-named scope 计数为零，仅说明没有命中该名称过滤器，不能声称通信等待为零。CPU collective scopes 可能嵌套，不能与 CUDA、host synchronization 相加。profile capture 包含 start/stop 开销、排除 trace export；与正式性能矩阵使用不同节点，因此没有隔离纯 profiler 扰动，也不据此推算端到端收益。

归因使用 CPU nesting 加 runtime correlation / External id，不按异步 CUDA 时间裁切。五个 scope 的归因分析中仍有 46259–60984 个未归属 kernel/trace（包括其他 DiT 工作或未能明确关联的事件），分析器将其归入 `ambiguous_or_unscoped_correlation`，没有将它们强行算入优化范围。14 份 trace 均包含目标 CPU scopes 和 CUDA kernels；详细状态、路由与未归属成本见 `correlation-and-routes.json`。

六份 profile 输入和输出 RGB 均与对应严格诊断完全一致，全部 360 帧完整解码；自动抽样观察沿用相同 SHA256 的两种输出；source 跟随画质的最终结论待人工验收，不能把旧自动观察称为人工判定未通过。profile 阶段未另存逐块 latent，不能将 RGB 一致称为 profile 阶段逐层数值检查。

### Trace 交付与 Perfetto

14 份来自 **2 个实现（eager/fused）× (SP1 的 1 个 rank + SP2 的 2 个 rank + SP4 的 4 个 rank) = 14**，对应 6 个 SP/variant profile 进程组，并非 14 次重复性能实验或要求评审者打开 14 个 UI。逐 rank 保留是为了检查非 rank0 的实际路由、kernel、通信与成本，避免只看 rank0 漏掉问题；各 rank 时钟未校准，因此不拼接成一条时间线。交付侧逐份导入 Perfetto 是检查文件可用性，阅读时可先看本报告汇总，再按需要打开某个 rank。正式 120 个性能样本不带 profiler，不与这些 trace 混用。

14 份原始 `.trace.json.gz` 与独立 `perfetto/` JSON 均已导出，未合并未对齐的 rank 时间线。总原始 JSON 为 4.724 GiB，压缩文件为 281.177 MiB；逐文件 phase/rank/大小/两种 SHA256 见 `trace-manifest.json`。分析器只读取原始单份文件，未将解压副本重复累计。

Perfetto UI `v58.3-11fbaed83` 已实际打开全部 14 份交付 JSON；UI SQL 中五个目标 CPU scope 的计数与离线分析完全一致，全部 CUDA kernel 计数逐文件一致，且每份都显示 CUDA stream 标签。每份有 1952–5783 条 `slice_spill_overlapping_complete_event` 显示警告：UI 说明重叠完整事件被移到 overflow track，未因此丢弃事件，但嵌套布局可能与原意不同。保留原始文件与警告，不改写 trace 消除提示。逐文件 UI 表格截图、代表性时间线截图和核对记录见 `perfetto-ui-evidence.tar.gz` 与 `perfetto-ui-validation.json`。这是实际打开与目标计数的验收，不是对所有其他事件种类、跨 rank 时钟同步或硬件利用率的证明。

## 复现命令与证据索引

完整实际 worker 命令、PAI 资源配置/Job ID/终态、观察器及分析器保存在产物根 `control-and-receipts.tar.gz`，内含逐文件 SHA256。`EXPERIMENT_RECEIPT.json` 标识实际执行的 runtime commit 和三个核心文件哈希。早期 `candidate-final` 包先于 tail-bit 修复；归档保留其历史名称，以后续 `fix-v3` 与 worker 实际导入哈希为准，不用早期包哈希冒充最终代码。

GPU 单测命令如下；预先导入模型是为绕过仓库现有 pytest conftest 导入环。CPU 作业另覆盖 window、experimental attention 与 model 测试，完整参数见 `cpu-r5.sh`。

```bash
python -c 'import sglang.multimodal_gen.runtime.models.dits.zing_h3; import pytest,sys; sys.exit(pytest.main(sys.argv[1:]))' \
  -x -vv python/sglang/multimodal_gen/test/unit/test_zing_history_rope.py
```

实际张量尺寸微基准通过 `HISTORY_ROPE_WORKLOADS` 指定冻结的四组片段尺寸；完整执行环境、资产验证、SP 顺序与服务启动参数见 `gpu-profile-r1.sh`。正式 120 样本与严格诊断的实际命令见 `gpu-full-r3.sh`，统计/媒体/原始数值导出见 `cpu-r21.sh`，trace 归因与导出见 `cpu-r22.sh`。这些是本次已执行脚本，含固定输出目录；复跑应准备独立输出根目录，不能覆盖本次证据。

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT=/tmp/history-rope-real-geometry \
HISTORY_ROPE_WORKLOADS=/path/to/actual-history-workloads.json \
SGLANG_LIGHTWEIGHT_RUNTIME=1 \
SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cuda \
python benchmark/zing_h3_causal/benchmark_history_rope.py
```

下表每行以首个完整目录为前缀，后续短文件名/子目录也位于同一目录；OSS 产物根见本文开头：

| 路径 | 内容 |
| --- | --- |
| `full-r3-final/analysis.json`、`samples.csv` | 全部 120 个原始耗时、12 个门槛、GPU 秒及输出一致性 |
| `full-r3-final/samples-per-rank.csv`、`memory-and-stage-per-rank.csv` | 逐 rank 阶段耗时及 allocated/reserved 峰值 |
| `full-r3-final/retention-manifest.json`、`numerical/`、`raw-rgb/` | 994 份原始数值证据、四份去重 RGB 及全部样本映射 |
| `full-r3-final/quality-review.json`、`full-r3-media.tar.gz` | 120 个样本对应的四种唯一输出审查、媒体及联系表 |
| `full-r3-final/full-r3-metadata.tar.gz`、`profile-r1-final/profile-r1-metadata.tar.gz` | 完整原始日志、配置、模块/硬件回执与观察器 |
| `profile-r1-final/micro.json`、`micro-summary.csv`、`micro-gates.json` | 48 条路径/尺寸组合的原始计时、成本与未通过的耗时波动门槛 |
| `profile-r1-final/history-trace-analysis.json`、`profile-scopes.csv`、`profile-ranks.csv` | 全部 scope 明细与逐 rank 成本 |
| `profile-r1-final/correlation-and-routes.json`、`actual-routes.json` | correlation 状态、路由与模块哈希 |
| `profile-r1-final/trace-manifest.json`、`traces/`、`perfetto/`、`profile-r1-traces.tar` | 14 份压缩原始 trace、独立 JSON 和 SHA256 |
| `profile-r1-final/profile-output-check.json`、`quality-review.json`、`profile-r1-media.tar.gz` | 六份 profile 输出核验及哈希关联的质量审查 |
| `profile-r1-final/perfetto-ui-validation.json`、`perfetto-ui-evidence.tar.gz` | 实际 UI 核对、截图及显示警告 |
| `aa-r1-audit/`、`full-r2-recovered/`、`full-r2-all-media/` | 未通过的原 performance 配置原始证据，不由新实验覆盖 |

## 最终验收结论

保持 `SGLANG_ZING_FUSED_HISTORY_ROPE=0`。本文原逐片段候选在独立确定性配置上通过同 SP 严格数值比较、全部正式端到端性能门槛及历史重建中位数改善门槛；原 production performance encoder 配置的严格数值重复性门槛未通过，11/12 组微基准对照的耗时波动门槛未通过；source 跟随画质已给出自动抽样观察和可播放对照，最终待人工验收。没有放宽数值容差、删除长尾样本或把 baseline 共有画质问题当作 candidate 合格。后续 grouped 路径的完整验收单列，PR 当前非 draft，没有自动合并，也没有停止其他人的作业。
