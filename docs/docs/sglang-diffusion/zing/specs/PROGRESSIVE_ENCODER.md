# #142 渐进 encoder 与因果缓存：接入契约 v1

状态：**本 PR 实现真实渐进 encoder 与会话缓存，仍在验收，不能宣称 #142 通过**。
依赖顺序为 #142 → #144 → #145；#131 负责整链路 24 FPS 与画质。
责任更正：#142/#146 负责真实渐进 encoder/cache；蔡逸文负责一般 Encoder/DiT 性能优化。
本任务不再等待外部 factory，不把重复编码前缀当增量实现。

## 实际代码审计

审计基线 `8962449043`：`MiniMaxH3VideoVAE` 固定 `EncoderFCN3D`，
`time_down=[1,2,2,1,1,1]`、`causal_encoder=True`、`use_t_isolated_gn=True`。
`BaseConv3d` 时间 padding 为左侧 constant zero；reflect 只用于空间。
`TemporalIsolatedGroupNorm` 将时间维合入 batch，统计不跨帧；fused 路径也必须保持该配置。
编码器没有 temporal attention，quant_conv 为 1×1×1。
因此候选最早稳定 latent 位置依赖 clip 内像素 0、4、8、12、16。
这只是依赖分析，不是 GPU parity 或性能证据；小时间维可能改变 CUDA 算法及舍入。

`_adaptive_encode` 的空间 tiling、重叠混合顺序、预处理和精度都必须保留。
每个 tile/layer 要保存足够历史特征、stride 相位与边界 padding；不可每次重算增长前缀。
旧路径每 17 帧重置卷积时间历史；新路径也在 17、34…重置，绝对索引不重置。
连续跨 clip 缓存是不同数学语义，须单列兼容性和画质 A/B，不能称 exact。

## 内核交付接口

提供可导入 factory `module:create_session(codecs, *, capacity_frames)`、完整提交 SHA、
checkpoint/config 支持范围，以及每层状态与计算量计数定义。返回
`ProgressiveEncoderSession`，沿用 `SourcePixelBuffer` 的 append/pop/finish/dispose：

- append 输入 uint8 `[3,T,480,832]`，连续绝对 `frame_start`；网络分包任意。
  容量不足或格式错误必须原子拒绝；消费输入后自行保留需要的存储，不借用调用者可变内存。
- pop 返回独占、不可回写的 `CodecSlice`，moments `[1,48,K,30,52]`，
  `[start,end)` 为绝对 latent 区间；返回 dtype 与旧 encode dtype 一致。
- **1+2+2 为 latent 输出数**。clip c 的完整输出依次是
  `[5c,5c+1)` / `[5c+1,5c+3)` / `[5c+3,5c+5)`；真实 source watermark
  是 `17c+1` / `17c+9` / `17c+17`（半开）。即输入增量候选为 1+8+8。
  水位不得偷换成当前 accepted_frames；多帧网络包也逐组 pop。
- 未 EOS 不得补未来帧；首帧 append 后必须 ready，pop 后已稳定。
  EOS 才能重复末帧到 17，保留 `1+ceil((tail-1)/4)` 个 latent，最后分组可短于 2。
  真实水位不计复制像素。空 EOS 拒绝；单帧 EOS 保留旧单图语义。
- 不要求 `pixel_end` 或 `consumed_frames` 扩展字段。消费方由绝对 latent end 与
  冻结的 clip17/ratio4 几何求 alignment end，只在 EOS 用真实总帧数截断；dependency
  独立保留。服务只对已提交 bridge 且成功完成调度的 alignment 范围释放完整 append
  credit，不按 encoder 缓存接管或 lookahead 提前释放（#144 最新对接约定）。
- pop 为 pull/backpressure；下游不读时 append 不得无限排队。
  `minimum_credit_frames` 至少 1 且不大于 8，必须允许逐帧推进。
- reset 丢弃未提交输入、缓存和相位，归零绝对游标；finish 后可 reset。
  dispose 取消并释放状态，幂等；dispose 后 append/pop/finish/reset 拒绝。
  两会话仅可共享只读权重。失败不得静默推进 cursor 或回写已交付 tensor。
- `state_usage()` 返回存储字节数和累计 temporal conv 输入位置计数（含重算），
  dispose 后 state_usage 仍可读且 retained_bytes 必须为 0；其他操作均拒绝。
  CUDA allocator 指标另采集，不能以 allocator reserved 冒充 cache bytes。

posterior moments 与采样分开，复用 `sample_streaming_visual_moments` 的
`indexed-cpu-v1`、绝对 latent_start、固定 posterior_seed。
这是分包无关 RNG，**不是旧 native 完整张量 seed 的逐位等价**。

## 冻结与验收

运行前冻结 model/config/权重、输入 RGB 与原视频 SHA256、源代码和 factory SHA、
精度、seed、RNG、SP、容差和缓存/计算上限。默认数值门槛 `atol=rtol=0`：
同权重同算子候选 exact，失败即保留失败，不根据实测放宽。
如需要 tolerance，须先声明新的独立 revision、误差依据与非 exact 属性。

验证器 audit 模式用真实旧内核比较 1/9/17 前缀与整 clip moments，
只探测可行性，明确是重复前缀计算，不可用于证明有状态增量或吞吐。
validate 模式必须显式提供真实 factory，不提供则不能获得候选通过结论。
1..34 帧尾部、120 帧多边界，逐帧、像素 1/2/2、固定随机分包、17 帧整块，
分别核对 moments、indexed posterior、shape、真实依赖、无重复遗漏和首输出水位。
像素 1/2/2 只是分包压力测试，与输出 1+2+2 协议不同。

正式性能固定 UniEdit v2 `uniedit-video-00007` 前 5 秒 120 帧，832×480、24 FPS。
旧 clip17 与候选各一次预热三次正式离线、三次 24 FPS paced；profile 单独运行。
保存逐输出时刻、输入水位、输入积压、同步 host encode、逐 rank 显存与原始 tensor。
当前验证器没有 DiT，因此首 DiT 可消费块与端到端画质/24 FPS 均保持未验证。
必须再检查 #144 scheduler 的真实门槛，匹配 checkpoint 四步配置，不能用 encoder 吞吐代替。

验证器还会用真实 adapter 检查无效 append 原子性、输入/输出存储归属、会话交错、
结束后/待处理输入时 reset、dispose 幂等和释放。
真实内核到位后还须运行长流的层级计数/缓存 plateau、
未来帧扰动、独立逐 rank trace；缺少任何一项不会关闭 issue。

## #144 接线审计（#140 的 `562e92e8a7a44bff2183cce08bf4e797051df01d`）

- `ZingH3SourceStreamAdapter.wait_for_next_chunk` 固定等待 `pending_frames >= clip`，
  `get_chunk_size`/`prepare_next_request` 也按 clip 派发；需要使用真实 encoder 推进粒度。
- `SourceChunkBridge.append` 用每次新增 pixel 数独立计算
  `1+ceil((pixels-1)/4)`，并拒绝 `accepted_pixels % 17 != 0` 后继续 append。
  因此第一组 1 latent 被当作最终短 clip，第二组必被拒绝；不能直接复用该输入校验。
- `source_session` 把 `encoded.source_dependency_end` 同时用作 `pixel_end`，
  并调用 `self.pixels.consumed_frames` 发起 ledger.consume。#144 必须区分对齐像素范围、
  真实依赖水位和可以释放的输入/ACK 水位；有 lookahead 的产物不能混用三者。
- `SourceReadySession.append` 严格要求覆盖 checkpoint 下一 chunk，EOS 才可缩短。
  第一 chunk 是否为 1 latent 必须读取实际 ChunkPlan，不得为降延迟改 checkpoint。
  本 PR 不改变上述服务代码；也不把本验证器首次返回的 moments 当作真实 DiT 启动。

## 命令与证据边界

在项目 NVIDIA CUDA 环境中运行（SP1，真实旧路径审计）：

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/validate_progressive_encoder.py \
  --mode audit --model /path/to/frozen-serving-package \
  --rgb /path/to/120-frame-832x480.rgb --output /path/to/new-run \
  --source-sha FULL_SOURCE_COMMIT --atol 0 --rtol 0
```

真实内核验收时改为 `--mode validate --adapter module:create_session`。
SP2 使用 `torchrun --standalone --nproc-per-node=2`，每 rank 单独写证据目录；
当前 H3 encoder 仍为 replicate，不能把 SP2 描述为 encoder 时间并行加速。
`frozen.json` 固定输入、权重、两个配置、代码文件、runtime、CUDA/PyTorch 与实测硬件。
`report.json` 始终保留未完成验收项，不生成全验收 PASSED 标记。
运行输入素材、原视频 provenance、完整源代码 archive 和其 SHA256 由 JobSpec 配套保留。

2026-09-21 首轮 GPU 审计固定提交 `40d5b540b94d092c4c224f70cc45cfd708c9520f`，
PAI Job `dlc14tqr62ojqre3`，结果目录
`/cpfs/world-model/sglang/evals/zing142-20260921/audit-sp1-r1/`。
后续验证器新增生命周期、未来像素扰动与 SP2 支持不计入该已冻结作业的结果。

2026-09-21 消费方对接更新：#144 已确认不需要新增 ownership/alignment 字段；
保留最小 CodecSlice 契约。ODE step1000 的 chunk plan 为 `[1,2,2]`、
`split_first_latent=false`（见 #142 最新优先级记录），真实运行仍需回读加载配置核验。

## 首轮真实权重结果（2026-09-22，北京时间）

`dlc14tqr62ojqre3` 已 Succeeded / exit 0，但数值检查**未通过**。
冻结源码 `40d5b540b9`，单卡实际报告 `NVIDIA L20D`、274113 MiB、CC 10.3，
PyTorch 2.13.0+cu130，BF16 encoder / posterior_seed 42。
原始 tensor 已下载回读并核对报告内 SHA256；没有修改零容差门槛。

| 对照 | moments max abs | RRMSE | 零容差结果 |
| --- | ---: | ---: | --- |
| 1 帧前缀 vs 17 帧的首 latent | 0.28125 | 0.00239149 | 失败 |
| 9 帧前缀 vs 17 帧的前三 latent | 0.125 | 0.00173894 | 失败 |
| 17 帧重复 vs 17 帧 | 0 | 0 | 通过 |

真实 tensor shape 为 `[1,48,K,30,52]`。以上不能证明数学非因果，也不能未经诊断就
全部归因于 CUDA 舍入；下一步保持完整17帧shape、仅扰动未来像素，独立核验未来依赖。
后续 audit 的 `--dependency-only` 可跳过已有计时基线，不重复申请性能矩阵。

旧 clip17 离线三次墙钟 4.06831 / 4.07137 / 4.06289 秒；同步 append/pop host
吞吐 29.514 / 29.491 / 29.552 FPS。paced 三次首 latent 为
1.16981 / 1.17194 / 1.17084 秒，输出间隔 p95 0.70909 / 0.70870 / 0.70967 秒。
峰值 allocated 11931280384 bytes（含加载的 VAE encoder/decoder）；不是纯 cache 大小。
上述仅是5秒旧路径基线，不是渐进encoder吞吐、长流不积压、DiT启动或整链路24FPS通过。

原始报告与 tensors 位于前述结果目录的 `audit/`；该首轮不包含新增未来扰动检查。

补充诊断 `dlc1wkvzeydiixpc`（源码 `877bb5f7f1`）也已 Succeeded / exit0。
保持17帧shape，分别将第1/9帧之后的像素替换为 `255-pixel`，首1/3个latent
均与原完整17帧编码逐位相同（max abs=0、SHA256相同）；已回读tensor独立复核。
1/9帧短前缀差异完全复现。这支持当前样本的因果性分析，不能单凭此诊断归因具体算子，
更不能宣称真实渐进缓存已达到 exact。该诊断未重复计时，`timings=[]`。

同一真实 `frozen.json.runtime` 已回读确认 `num_frame_per_block=[1,2,2]` 和
`split_first_latent=false`。当前两个作业均终态，无活动GPU；不再重复旧路径诊断，
由本 PR 实现真实有状态内核后继续验收。原始补充证据目录为
`/cpfs/world-model/sglang/evals/zing142-20260921/dependency-sp1-r2/audit/`。

## PR #149 集成后的验证矩阵（2026-09-22）

已合并 main `e4697e920f7fc4ce544a228f617ec33193434e79`，集成提交
`977ae1f68d204824e1bb5b62787f9ab325029aaa`，没有重复 cherry-pick #140/#146。
#149 使旧 clip17 与 streaming moments 遵循实际 `encode_mode`；它没有提供渐进 factory。
旧 r2 loaded runtime 已确认 ODE step1000 的 `[1,2,2]`、`split_first_latent=false`、
BF16、`encode_mode=performance`，但旧源码强制 deterministic 后端策略。
因此旧 r1/r2 证据保留原样，不能标成 #149 修复后的 performance 验收。

| 项目 | 本轮范围 / 后续门槛 |
| --- | --- |
| CPU 回归 | 渐进契约、media contract、streaming codecs、model/export；fixture 仅验证接口，不证明真实缓存 |
| 新候选冻结 | 最终完整 SHA、factory SHA、step1000 权重/config hash、实际 loaded runtime 与后端策略 |
| 输入与调度 | 832×480 / 24 FPS，`[1,2,2]`、split_first=false；DiT few_step 显式 `[1000,750,500,250]` |
| 数值与功能 | 旧 clip17/performance 同策略对照；atol=rtol=0；moments/indexed latent、1/9/17 依赖、跨 clip/EOS、生命周期 |
| 长流性能 | SP1 后 SP2，10 秒 / 60 秒 paced 和离线吞吐、缓存上界与累计计算量；独立正确性、无 profiler 计时及独立 trace |
| 本方实现 | `progressive_encoder:create_session`，真实逐层卷积历史与下采样相位；待 CPU/GPU 数值及资源验收 |

本轮不提交 GPU、不重复旧前缀诊断。新组合版本的 GPU loaded config、四步 DiT 执行、
10/60 秒测试均未执行；仅有旧 loaded config 回读与集成后的 CPU 检查不能替代它们。

责任边界以最新用户更正为准：上文“交付接口”是 #146 对 #144 的交付要求，非外部阻塞。
#149 集成回归：618 passed；首次隔离导入方式与 unit conftest 冲突，改用正常包导入后通过。

## 真实缓存实现首版

最终集成与验收统一在 #148，#146 仅保留历史，不重开。
Factory 为
`sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder:create_session`。
`CausalEncoderCache` 不修改共享 forward，不重算增长前缀；每层保存 kernel temporal
历史及已见位置，固定 1/8/8 输入推进对应 1/2/2 latent。逐 clip 清空历史，绝对索引连续。
使用原生空间切片、stack 顺序和纵向后横向 blend；encoder SP 按复制模型执行。
权重需在创建/使用会话前由加载器准备为配置 dtype；不在活跃会话中转换共享权重。
当前支持标准无条件 EncoderFCN3D、时间隔离归一化、因果零 padding、clip17/ratio4。

37 项真实小型 CNN CPU 检查通过，包括 1–34 EOS、跨 clip、两会话隔离、reset/dispose、
失败重试及每 clip 固定工作量/缓存上界。Float64 数学对照使用 1e-12，仅属 CPU 算法检查，
生产 BF16 仍使用原 atol=rtol=0。训练权重、真实 tiling、SP1/SP2、10/60秒尚未验收。
首个测试配置空间尺寸不足以通过两层无空间 padding 的 downsample；增至 8×8 后通过，
没有改变内核或生产阈值来规避该测试输入错误。

补充 CPU 证据：41 项真实小 CNN 检查 + 46 项契约检查 = 87 passed。
新增原生空间 tiling/纵横 blend、stack 开关、zeros/reflect 与逐帧/整17/混合分包；
分包间输出逐位相同，Float64 完整 clip 对照仍仅为 1e-12 算法检查。
验证器在建立任何会话前显式准备 encoder/quant_conv dtype，并写入 frozen.json，
不再依赖 baseline encode 的副作用。最终组合版本 GPU 提交统一由 #148 任务负责。

## 后续 revision 的覆盖补齐

- validate 模式新增真实 candidate 的未来像素扰动：完整17帧分包输入，分别只改
  第1/9帧之后像素，首1/3 latent 与未扰动 candidate 零容差比较。单独保存三组原始
  candidate tensor；这与旧 encoder audit 分开记录。
- 生命周期数值比较前保存 actual/expected tensor 及逐会话误差 JSON，避免首个
  lifecycle 失败仅剩错误字符串。门槛不变，仍 fail-fast，不将未执行项视为通过。
- 这些功能不属于已冻结 f0eec536 SP1 作业。10/60秒、长流资源门槛、逐层误差定位、
  独立 trace 和整链路检查仍缺失，不因新增未来扰动而标记完整验收。
