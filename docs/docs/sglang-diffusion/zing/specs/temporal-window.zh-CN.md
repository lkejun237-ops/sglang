# Zing temporal chunk window (#125)

本次集成基于 main `f1428c6d8616cdb4bc61b6fe81e10ee571f58e21`，保留 #121 的失败关闭语义和 #126 的共享 scheduler 实现。#127 冻结分支的回滚测试属于历史证据，不作为本次验收依据。

## Policy

Defaults `zing_window_chunks=21`, `zing_sink_chunks=3`, `zing_rope_position_mode="block_relative"`, `zing_rope_max_frame_gap=12`: three sink chunks, seventeen recent chunks and the current chunk. A source/target/audio pair counts as one temporal chunk. Prefix/reference tokens are separate. IDs remain absolute: at chunk 21, ID 3 is evicted; at chunk 29, retained IDs are 0,1,2,12,...,29.

Both scalar 5 and repeating [1,2,2] chunk plans preserve checkpoint geometry. GPU acceptance uses [1,2,2] with the AR checkpoint; it does not establish DMD quality.

## Positions and ownership

Pre-RoPE K and original checkpoint positions are retained. Every forward reconstructs attention-only positions and rotates all visible K, including history. Absolute chunk IDs, source provenance and pixel coordinates never change. Prompt updates rebuild only the prefix.

The `block_relative` profile adapts MinWM gap compression to Zing's repeated source/target/stereo-audio timestamps: sort the unique visible **media** time coordinates, retain equal coordinates across roles, cumulatively sum gaps capped at 12 checkpoint temporal units, and anchor at the end of the unchanged prefix/reference coordinates (zero without a prefix). Spatial coordinates remain unchanged. This is not MinWM's unique-consecutive token-order mapping, which would move backwards when Zing revisits a timestamp. The gap is an explicit inference profile, not a checkpoint-trained guarantee; it is not measured in pixels or temporal chunks. SP ranks construct the global map before slicing queries.

Explicit reference configuration: `zing_window_chunks=null`, `zing_sink_chunks=0`, `zing_rope_position_mode="absolute"`, `zing_rope_max_frame_gap=12`. CLI: `--full-history --sink-chunks 0 --rope-position-mode absolute`. Finite-window/absolute is also supported for isolating the RoPE effect. Relative versus absolute may change generated content; no exact equivalence is claimed. Same-policy finite/incremental and SP1/SP2 comparisons are separate gates.

The incremental position grid materializes only the current span, preserving sequential FP64 rounding. Arithmetic remains O(absolute offset); this is not a constant-time infinite-session claim.

History selection occurs before source execution. Any source, active, final or post-commit failure closes sampling state and disposes SourceReadySession; subsequent step, append and prompt updates are rejected. Caller RNG is preserved. Cursors advance only on success; cancellation and failure are not successful EOS.

| Owner | Bound |
| --- | --- |
| KV and positions | 21 committed temporal chunks plus independent prefix; at most 20 history chunks during current execution |
| Ready tensors | max_pending_chunks and max_pending_bytes; append copies views into owned storage |
| Provenance | max_pending_spans (default 4096); completed output history is not retained |
| Layout and positions | Current chunk and retained KV; absolute cursors continue growing |
| Prompt/reference | Pending/applied snapshots; size depends on request, outside the KV budget |
| Codec and network | Separate ownership and budgets; this feature does not prove their boundedness |
| Finite sampler/adapter | Full finite source/noise/output/layout remain externally owned; only KV is bounded |

## Acceptance

CPU 回归作业 `dlcxscv0qxiq0yc8`：325 passed，21.83 秒。代码固定为 `13f3ccb53b1a70aab6f8c9e2442d7506d451f4f0`。日志已从 OSS 回读，SHA256 为 `e67821b79a7d893606c9a246237c740bfe470dd6f20278bdaace30488aea457e`。

SP1/SP2 真机正确性与生命周期验收已通过并回读工件：44 chunks 的同窗口有限/增量输出逐值对比、保留 ID、首次淘汰后的失败清理、内存平台期、取消/EOS 释放。性能 A/B 另见性能报告。

当前测试环境为 Torch 2.12.1 / CUDA 13.0。main 强制依赖的预编译 kernel 与该环境 ABI 不兼容，因此使用同一源码构建匹配版本，未恢复可选依赖回退。构建作业 `dlcxipwqyc0cv9q9` 已成功，wheel 已回读核验，SHA256 为 `f1a8272dedb35f87b239de1cb157b434f97969cc8c62d3fcde6870fbe417b053`。SP1/SP2 均通过 GPU kernel 运算预检后执行验收套件。

本次验收不包含独立 native 画质对比、窗口与 full-history 感知质量等价或 WebSocket 断连清理。有限与增量路径共享 DiT/采样逻辑，exact 比较不能排除共享实现错误。#127 的冻结版本结果不作为本次 main 集成的证据。

长窗口正确性和性能 A/B 使用同一组输入，均从工件中的 `input-manifest.json` 核对：

| 输入 | SHA256 |
| --- | --- |
| Source | `d96c3e2dbdd62a1e0d11c164767b34030eb6b3cf49e417127930691feeeacc79` |
| AR checkpoint | `66eb0b047586a4133038cebab5e54cc71c38a43d325a6807fc6739b23f9b1c5e` |
| Runtime config | `09861d3ae5aa01d2ea61a2f5af24c2915b1189d86c0bf2bab489d02eb1729c30` |
| Request | `375fad262e9ef0ebac0488e3a1f4b77addd2325b2bdb8419b0454856c10d03cf` |

生命周期定向测试使用同一权重、22 帧短请求；输入与长窗口请求不同。每卡实测 NVIDIA L20D、274113 MiB、CC10.3，未按产品名称推断显存容量。

### SP1 已回读结果

作业 `dlcwf9slmftao1ob` 的长窗口阶段与生命周期阶段均 exit 0。44 chunks 的视频/audio latent 逐值一致，首次淘汰后失败自动关闭、释放并拒绝重试。末段 10 个未结束 chunk 的 GPU allocated/reserved、host RSS、KV storage 跨度均为 0 bytes；EOS 会话持有计数全部清零。输出顺序 SHA256 为 `f50dd6c04e59101e72beefc0cc8823113e070cbeeb3891c7c47e90a00ef16eda`，240 帧媒体完整解码通过。

pending cancel、committed cancel、terminal exception、successful EOS 四类路径均释放全部 owned tensor weakrefs，GPU allocated 每次精确回到 `137292077056` bytes。工件前缀为 `oss://leap-worldmodel-thailand/world-model/sglang/evals/zing-h3-dmd/zing125-window-main-sp1-r5/` 与 `zing125-window-main-lifecycle-sp1-r5/`。整个作业已 Succeeded，三个阶段 exit 0。

### SP2 已回读结果

作业 `dlc1nw8emivca5kp` 的长窗口与生命周期阶段均 exit 0。两卡各 44 chunks 的视频/audio latent 逐值对比、保留 ID、首次淘汰后的失败关闭与释放均通过；末段 GPU allocated/reserved、host RSS、KV storage 跨度均为 0 bytes。两卡输出摘要与 SP1 相同。独立落盘的 SP1/SP2 visual latents `[1,24,72,32,56]` 与解码音频 `[2,320000]` 逐值一致，max abs 和 RRMSE 均为 0；媒体完整解码通过。

两卡各四类生命周期路径全部通过，owned tensor weakrefs 全释放，GPU allocated 每次精确回到 `137357364736` bytes。工件前缀为 `oss://leap-worldmodel-thailand/world-model/sglang/evals/zing-h3-dmd/zing125-window-main-sp2-r5/` 与 `zing125-window-main-lifecycle-sp2-r5/`，包含 cross-sp 和逐 rank 报告。整个作业已 Succeeded，三个阶段 exit 0。

### 性能结论与受测版本

固定 240 帧 / 44 chunks AR 请求，每种配置预热一次、正式三次取中位数。SP1 整体 FPS 从 1.6917 提升到 1.9728（+16.62%）；SP2 从 2.6414 提升到 3.0062（+13.81%）。采样阶段吞吐分别提升 29.88% 和 24.42%。性能工件前缀分别为同桶的 `zing125-window-main-perf-sp1-r5/` 与 `zing125-window-main-perf-sp2-r5/`，完整方法、原始耗时和 SHA256 见配套 PR #129。

性能只适用于该固定 AR 用例，不能解释为画质等价或 24 FPS 实时达标；scalar5 对此长度仅产生 15 chunks，不会触发 21 块窗口淘汰，不能套用上述增益。

真机受测运行时代码固定为 `13f3ccb53b1a70aab6f8c9e2442d7506d451f4f0`；后续本分支提交仅修改文档。测试期间 main 新增 `f45c54783c88dd473d90386509d3dfdb4cf30cfd`（#133，默认关闭的逐卡内存采集）。已核对差异、无冲突合并结果和合并后验证脚本语法；该合并结果未重新跑真机，不把本次结果标记为最新 main 合并提交的完整重测。

## Relative RoPE 验收记录（#132，2026-09-21）

实现基于 #130 的 `13f3ccb53b`；GPU 执行代码冻结在 `253a17500f`，之后的提交只更新文档。PR 提交链已重放到 #130 当前的 `1864b2f350`；`13f3ccb53b..1864b2f350` 只新增基线验收文档，排除本文档后，重放前后运行时工作树逐字一致，因此没有重复提交 GPU 作业。#127 实际合入 `codex/zing-window-125-base`，没有直接进入 main。当前实现保留 main 的失败即关闭、清理、拒绝重试语义。

### 环境和 CPU 回归

- 最终零 GPU 回归 `dlc4xvqbusnc69c4`：343 passed、21 warnings、28.04 秒。覆盖 absolute 与 block_relative 的同策略 finite/source-ready 对照、标量 5 与 `[1,2,2]` AR 分块、首次与持续淘汰、sink/no-sink 百万级绝对偏移、prompt 更新、失败清理、EOS、取消和 CPU SP。
- 运行环境为 Torch 2.12.1+cu130、CUDA 13.0。现有 kernel wheel ABI 不匹配，因此使用同一镜像内 `/usr/local/cuda-13.0/bin/nvcc` 从当前源码构建 `sglang_kernel-0.4.6.post1+torch2121cu130-cp310-abi3-linux_x86_64.whl`；SHA-256 为 `473a07174f29f7cafb28773f5d60262195cc4849bcfa5cfd6508d461d66deca9`。这是运行依赖修复，没有修改 Encoder/DiT 算法。
- 所有 GPU 作业先读取实际设备属性：NVIDIA L20D、274113 MiB、compute capability 10.3。由此选择 DiT resident、encoder replicate、VAE/text encoder 不 offload。

### 10 秒 SP1 → SP2

输入为 832×480、24 FPS、240 帧，AR chunk plan `[1,2,2]`，four-step `1000/750/500/250`，共 44 temporal chunks。两项作业都使用窗口 21、sink 3、`block_relative`、gap 12。

| 模式 | 作业 | 结果 | 生成时间 | 峰值显存 |
| --- | --- | --- | ---: | ---: |
| SP1 | `dlc805b7uaep2mbh` | Succeeded / exit 0 | 273.42 s | 173044 MiB |
| SP2 | `dlchpwpvu70gmjgo` | Succeeded / exit 0，rank 0/1 均通过 | 251.68 s | 164600 MiB |

- finite-window 与 source-ready incremental-window 在两个 rank 上视频 latent、解码音频 exact；SP1 与 SP2 的 `output_sha256` 同为 `968785b90e1901b2815f311db3043298b2b9bf684c3e1fe9a1bed20ebe361023`。
- chunk 21 的 retained IDs 为 `0,1,2,4,...,21`；之后始终为三个 sink、十七个最近历史和当前 chunk，EOS 后为空。首次淘汰后的注入异常正确关闭会话。
- 稳态 `gpu_allocated`、`gpu_reserved`、host RSS 和 KV storage 观测跨度均为 0；EOS 后 ready、provenance、KV 与 static storage 全部归零。完整有限输入、完整输出、静态 prefix/reference、codec 和网络队列不包含在该窗口边界内。

### 生命周期释放

SP1 `dlc1dwlma2j6io39` 和 SP2 `dlc6cizgqcjq7aq5` 均 Succeeded / exit 0；SP2 rank 0/1 都通过。`pending_cancel`、`committed_cancel`、`terminal_exception`、`eos_success` 四种模式均释放所有 owned tensor 引用，释放后 ready/KV/static 计数为 0，GPU allocated 回到测试前数值。此处只验证 GPU `SourceReadySession.dispose`；WebSocket 路径属于 #131。

### 60 秒窗口长测

输入为不同真实片段不循环拼接的 832×480、24 FPS、1440 帧技术样本，共 255 temporal chunks。它验证长时窗口、SP 和资源释放，不作为连续单场景画质证据。

| 模式 | 作业 | 结果 | 生成时间 | 输出 SHA-256 |
| --- | --- | --- | ---: | --- |
| SP1 | `dlc6cerre6uptjdf` | Succeeded / exit 0 | 1251.77 s | `f802f6a8133138f4e4aec9212cf43a783523151a76c1134fb38d4a7b720e4331` |
| SP2 | `dlc17t5seivagmgr` | Succeeded / exit 0，rank 0/1 均通过 | 914.69 s | `f802f6a8133138f4e4aec9212cf43a783523151a76c1134fb38d4a7b720e4331` |

SP1、SP2 两个 rank 的视频 latent、解码音频 finite/source-ready 均 exact，跨 SP 输出 hash 也一致。两种 SP 的 GPU allocated/reserved、host RSS、KV storage 稳态跨度均为 0，EOS 后全部窗口持有量归零；峰值显存分别为 173824 MiB 和 167364 MiB。

### UniEdit v2 质量对照

UniEdit v2 清单包含 64 个 source-video 用例，目标最多 132 帧；镜像媒体最长 5.504 秒。质量对照固定为 `uniedit-video-00007`：125 帧、832×480、24 FPS、相同权重/seed/four-step，分别运行窗口相对、窗口绝对和全历史绝对配置。三项作业 `dlc1oq5s7eeaqv9o`、`dlc1ri1q16bp0i61`、`dlc1v3wg0ano9dwd` 均 Succeeded / exit 0。

- 骆驼从首帧出现并持续到结尾，编辑指令得到执行；三种策略没有可辨识的相对策略特有退化。
- 窗口相对与窗口绝对从帧 78 开始有差异，整体 PSNR 51.584 dB、SSIM 0.997447；与全历史绝对从帧 77 开始有差异，PSNR 51.692 dB、SSIM 0.997114。它们接近但不是 exact。
- 三种策略共享明显质量问题：女人后半段被骆驼遮挡并发生主体融合，尾段有色彩/对比度漂移。因此单例“相对策略不劣化”通过，但成片发布质量未通过；该 AR step 3500 结果不冒称 DMD 质量。
- 可复核报告：[UniEdit v2 四宫格、分项视频、关键帧与运行数据](https://review-cdn.loopit.com.cn/reports/zing-h3-relative-window/20260921-r2/index.html)（办公网或公司 VPN）。报告清单中的 15 个对象已从 OSS 按字节数和 SHA-256 回读；CDN HTML 200、视频 Range 206、MIME 正确；Microsoft Edge 已实际加载页面并播放到尾段。

媒体准备作业 `dlc10kimtmy3w1li` 同时生成 240/1440 帧技术输入和上述原片编辑请求；manifest 记录原始 URI、文件 SHA-256、片段顺序、处理后文件 hash 与请求。relative、window-absolute、full-history-absolute 之间只做画质比较，不将策略差异报成数值回归。
