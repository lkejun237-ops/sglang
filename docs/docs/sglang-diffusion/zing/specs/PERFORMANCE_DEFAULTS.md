# Zing 性能优化默认配置

B300/H200 部署现在可使用包内 `zing_h3_profile_launcher`，通过共用配置设置加速环境变量、
默认开启 Qwen 六项执行选项及 prompt-joint，两种硬件均使用 quantum=1 与流水线重叠，并支持独立 Encoder/Decoder 的对应配置；
配方、覆盖规则和当前未实测边界见
[部署 launcher](../deployment_zing_h3.mdx#b300--h200-统一-launcher)。
下表是历史源码默认值记录，不能替代 launcher 的有效环境回执。

基于主线 `df69ad4265` 梳理最近合入的优化。下表环境变量均省略
`SGLANG_ZING_` 前缀（单独注明者除外）。显式设置仍优先于默认值。
Window/Sink 默认值于 2026-11-03 更新为 `21/2`，对应当前 unsplit `[3,2]`
模型的 5-latent 锚点；历史 W6/S3、旧 split-first 的测量仍保留各自配置。

Source 流水线默认采用 quantum=1、cross_chunk、async output、defer-final、
prompt-joint，Zing 校验后的 realtime 会话容量为 1（公共默认保持 8）。remote Qwen 默认 conditioning prefetch，
remote Encoder 默认 overlap + prefetch，独立 remote Decoder 在 quantum=1
时默认 overlap；本地组件保留本地路径。布尔开关可显式传 false，详见
[流水线默认值与兼容边界](../plans/PIPELINE_DEFAULTS.md)。
Block0 cache 沿用 #455 的默认 on 和 host/disk 字节预算，旧 overlap/entries
环境变量已移除，不应继续配置。

| 字段 | 默认值 | 用途 |
| --- | --- | --- |
| `FUSED_HISTORY_ROPE` | `1`（本次开启） | 融合历史 K RoPE；不支持的设备和布局仍回退 |
| `FUSED_CURRENT_ROPE` | `1` | 当前块 Q/K RoPE 复用融合历史 RoPE kernel，支持 A2A 后交错 head stride；固定槽优先走 RoPE/KV 融合，其余走此入口；不支持的设备和布局回退 |
| `FUSED_CURRENT_ROPE_KV` | 已验证 480p 固定槽路径为 `1`；其余为 `0` | A2A 后融合当前 Q/K RoPE 与 KV 写入；`0` 回退当前 Q/K RoPE 分发 + index_copy，`1` 显式尝试受支持布局 |
| `FA4_FIXED` | 已验证 480p 固定槽路径为 `1` | 最佳分段 FA：Q390/780 使用 fixed Q + GPU KV 有效长度；Q1170 保留 varlen；`0` 回退全 varlen |
| `GROUPED_HISTORY_ROPE` | `1`（本次开启） | 多片段 grouped 处理；`contiguous` 是替代模式 |
| `CHUNK_ROPE_REUSE` | `1`（本次开启） | 按 chunk 复用未失效的历史 RoPE |
| `INITIAL_PREFIX_JOINT` | `1` | 首次未命中时联合 condition/source/target step0；`0` 仅恢复独立 prefix + source/target 联合，仍受视频/joint mode/布局准入限制 |
| `HISTORY_ROPE_PLAN` | `0` | 跨层复用 forward metadata；现有实测无稳定额外收益 |
| `EXPERIMENTAL_ATTENTION` | `fa4` | 已默认使用 FA4 |
| `EXPERIMENTAL_STABLE_SHAPES` | `1` | 已开启稳定形状 |
| `EXPERIMENTAL_COMPILED_ADALN` | `1` | 已开启编译 AdaLN |
| `EXPERIMENTAL_FA4_METADATA_REUSE` | `1` | 已开启 forward 内 FA4 长度元数据复用 |
| `DENOISE_CUDA_GRAPH` | `1`（本次开启） | 用 CUDA graph 回放历史 RoPE 全部复用的 denoise forward，与 eager 逐位一致；仅 SP1 + KV 池，其他场景自动走 eager；预热失败回退 eager；捕获失败会停用 graph 并报错，建议重启进程 |
| `DENOISE_CUDA_GRAPH_MAX` | `32` | 每个模型保留的 denoise graph 上限（LRU）；多会话时 key 含各自的 KV 槽地址 |
| `MINIMAX_H3_PROGRESSIVE_ENCODER_TILING`（完整名称） | `auto` | 渐进 source 编码按已有条件自动选择 tile 并行 |
| `MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST`（完整名称） | `1` | CUDA/BF16/performance/tiled source Encoder 采用局部 channels-last 布局；`0` 恢复原布局，其他准入条件保持不变 |

Q/K fused RMSNorm、store-backed KV / fused KV write、移除 DiT / sampler
host sync 已直接集成，无需额外开关。已退休的 `EXPERIMENTAL_CACHED_ROPE`、
`EXPERIMENTAL_KV_APPEND`、`EXPERIMENTAL_KV_SINGLE_STORE`、
`EXPERIMENTAL_FUSED_ROPE_KV_WRITE` 不应重新设置。

## 预分配 KV 池的容量字段

通过服务参数 `--mem-fraction-static`（`mem_fraction_static`）启用静态 KV 池。
该比例与 LLM SGLang 一致：静态预算包括常驻模型及 KV，其余显存用于运行时计算。
`None` 保留未启用池的路径；通用默认比例与完整部署组合仍待同硬件验收。
历史合并版本的 H100 独立 DiT 配方见[部署 recipe](/docs/sglang-diffusion/zing/deployment_zing_h3)，不作为所有执行组合的默认值；当前基于 main `17a7ac4835` 的容量提交未新增 GPU 验证。

启动时在模型加载前记录实际可用显存 B，模型、Encoder 与 Lynn 启动预热后执行
`gc.collect()`、同步及 `empty_cache()`，取得空闲显存 F：

```text
ratio reserve = ceil(B * (1 - mem_fraction_static))
additional reserve = max(0, 8 GiB - ratio reserve)
KV bytes = floor(F - B * (1 - mem_fraction_static) - additional reserve)
bytes per token per rank = 3 * layers * local_heads * head_dim * dtype_bytes
total tokens = min_across_ranks(KV bytes // bytes_per_token)
```

三个存储分别为 K、V、rotated K。只分配一个 slot，要求
`realtime_max_sessions_per_worker=1`。prefix、source、target 共享总容量；当前服务计算边界仍拒绝音频生成，
不再提供 pool size、prefix 或目标画幅对应的独立容量参数。
启动日志 `ZING_KV_CACHE_POOL_READY_JSON` 保留逐 rank 的加载前/后空闲字节、
ratio、KV 字节预算、总 token 数、实际池字节数及最低／额外／跨 rank 共同运行时预留。

KV 行数容得下请求，不代表 DiT 临时分配也容得下。2026-11-04 的 H100 实测中，
`87078a671d` 在 38,614≤40,147 行时仍于 FFN 激活分配 OOM。当前修订同时检查
`3 GiB + 64 bytes × max(local_hidden_width × forward_rows, full_hidden_width × prefix_rows)` 的计算空间预算；
`forward_rows` 覆盖 prefix/source/target 首次联合计算、跨 chunk 的前一块 final，
以及 prompt 替换的 clean history。串行重建取阶段最大值；启用 prompt joint 时计入同次
forward 的 prefix、clean history、source 与 target。稳定形状模式的独立前缀预填充与串行重建
会在每个 SP rank 复制完整前缀、使用完整 hidden width；只有保存的 KV heads 分片，
不能用局部 head 宽度估算该阶段。`c8e6049c34` 的 H100 R33 已实际暴露该漏项：
原始 21 图请求在 prompt 切换重建时申请 672 MiB，而 driver 仅余 242.62 MiB。
修订后的前缀包络已在 `9b7084342d` 的 H200／SP2 安全场景中验证；KV 能容纳的输入仍可能因计算空间不足被明确拒绝。
`696acf0da6` 的 H100／SP2 真实输入 DiT 专项另验证了两档最大产品输入、换图及拒绝后恢复，历史受测配方建议 ratio 0.85；当前新主线提交未新增 GPU 验证，H100 完整链路、B300 及其他执行组合不能由此外推。
`679aefa5a4` 合入主线 worker readiness／故障清理修复后，容量公式与 DiT 计算路径未变；
原安全结果保留 `9b7084342d` 的版本边界，合并后完整服务 GPU 组合未重测。
缓存恢复同样检查当前会话的执行模式，
不会因预填充时关闭 joint 而跳过更新时的检查。各 rank 使用共同的最小运行时预算。
此空间包络与 8 GiB 下限是保守策略。H200／SP2 已验证原始最大输入及拒绝后恢复，尚不构成任意硬件、模型或后端的推荐值；证据必须保留原始最大输入、
480／736、低比例与接近 1 的比例、冷启动／Graph／缓存／切换的实测及拒绝后恢复证据。
更换架构、精度、attention backend 或联合执行模式时需重新验证，不能外推当前测试结果。

请求初始化及更新先在 CPU 元数据层检查计算输入边界，packing 后再用现有
`peak_rows()` 检查完整请求。画幅、图数及 Qwen 条件行数是计算边界，
不能当作已预留的 prefix 配额，也不代表所有组合均已完成显存验证。

当前静态池的编码前计算边界：最大画幅 1280×736（仍受 checkpoint 短边及网格约束）、
UTF-8 prompt 最多 65536 字节、Qwen condition 最多 16384 行、参考图数不超过部署图数限制，
参考/呈现媒体仅接受单帧图像，单张最多 1,032,192 像素（checkpoint 的默认参考图预处理边界），
不复用输出画幅面积上限。source 使用视频编码边界：短边不超过 480、长边不超过 1280、
面积不超过 832×480；请求中的面积覆盖和已准备的缓存输入也不能绕过此限制。
有限请求的输出与 source 均不超过 720 帧，
尚未验证的音频输出在编码前拒绝；连续 source 按已有有界队列处理，不按完整会话时长分配。
这些边界限制危险分配范围。`9b7084342d` 的 H200／SP2 完整安全验收覆盖多图、两档输出、低／高比例、切换与缓存、旧／新状态共存及拒绝后恢复；详见[部署 recipe](/docs/sglang-diffusion/zing/deployment_zing_h3)。按用户要求，性能补测收敛为 DiT-only，完整 50 层两档的独立数值回读已通过；H100 的受测 ratio 建议不替代真实输入全链路性能、通用默认比例及 integrated 默认组合验收，不将专项结果外推为上述范围通过。

### 输入策略的产品行为与依据

以下是本轮静态池的保守服务策略，不是任意 checkpoint／硬件的物理极限，也不是“允许输入”就必然有 KV 容量。
启用静态池的普通入口在编码前检查；独立 Encoder 即使不知道对端池容量，也独立限制编码几何；
准备缓存恢复时在重新复制到 CUDA 前再次检查元数据，不能绕过相同边界。

| 边界 | 用户可见行为 | 依据与验证范围 |
| --- | --- | --- |
| 输出面积≤1280×736、长边≤1280，且符合 checkpoint 短边／空间网格 | 超限尺寸在编码／分配前拒绝 | 冻结的 480／736 输出包络；网格和短边由 checkpoint／packer 决定，不能外推其他画幅 |
| source 短边≤480、长边≤1280、面积≤832×480 | 过大的 source 编码输入拒绝；允许 source 480、输出 736 | 视频 source 与单张参考图走不同的计算路径；该包络来自已冻结 source 工作负载，不以较大的静态参考图包络证明视频编码安全 |
| 有限 source／输出≤720帧 | 有限721帧请求拒绝；连续 source 按有界块队列处理，不将完整会话长度当一次分配 | 显式限制一次性有限输入的 CPU／VAE 分配；这是当前服务的保守准入策略，不是模型只能生成720帧，完整长有限输入需另行验证 |
| prompt≤65,536 UTF-8字节，Qwen条件≤16,384行 | 长文本／图文编码超限拒绝 | 字节边界保护 tokenizer／编码前资源，行边界独立约束准备结果；两者不是同一量，不能由字节数精确换算token，也不据此修改原始prompt |
| 参考图数≤`zing_reference_max_images`，每项仅一帧、短边≤checkpoint上限、单张≤1,032,192像素 | 视频参考或超大参考在编码前拒绝 | 图数来自部署声明；面积沿用默认 checkpoint 的静态参考预处理包络，不能将它当成 source 视频预算。source presentation 不计入参考图数，但仍检查其单帧几何 |
| video-only输出 | 请求音频生成时明确拒绝 | 当前静态池的编码／工作区包络未验收音频，不将旧音频计数能力当作安全支持 |

例如 source 832×480、输出1280×736仍是合法几何，但其真实条件经packing后还要检查共享KV行数及计算工作区；
原始15／21图输入可能通过编码边界，随后被容量或工作区拒绝。初始拒绝只结束该请求，后续会话可恢复；
更新拒绝保留当前active条件与KV，后续合法更新仍可生效。错误使用稳定ASCII code和友好英文消息，详细形状／字节仅写服务端日志。
本次迁移不依赖PR #584，没有放宽这些策略，也没有新增GPU实测；历史合并版本的结果和具体 source SHA 见部署 recipe／PR #517。

有限与连续两条 realtime prompt 更新路径共用精确 packing/admission；更新还模拟五帧 clean target history 重建，及重建后首块在窗口淘汰前的 source/target 峰值，取其与普通生成峰值的较大值。小窗口不能用普通稳态容量替代切换容量。
候选准入成功前不会安装 pending condition 或清理当前 KV。准备缓存命中在恢复 GPU tensor 前检查计算元数据与对应的初始／更新容量；初始身份由接收方会话状态判定。

| 窗口字段 | 默认值 |
| --- | --- |
| `zing_kv_pool_window_chunks` | `21` |
| `zing_kv_pool_sink_chunks` | `2` |
| `zing_kv_pool_rope_position_mode` | `block_relative` |
| `zing_kv_pool_rope_max_frame_gap` | `12` |

当前请求默认 window21 / sink2、block_relative、max_frame_gap=12；
解码模式仍为 `stable`。修改窗口或改为 stateful 解码会改变请求语义，不属于纯算子开关。
请求与 KV pool 的启动 window/sink 必须一致。显式历史 S3 profile 保留原值；
动态 `sink.off` / `sink.refresh` 仅支持 unsplit `[3,2]`、W≥4 和启动 S0/S2，
详见 [动态 Sink 规格](dynamic-window-sink.zh-CN.md)。

选择 prompt/reference 预算时，可参考 [H200 两卡输入预算](/docs/sglang-diffusion/zing/prompt-reference-budgets)。
该页基于历史 W6/S3、拆分首块、Graph 关闭的 DiT 实测；25.6k 是约20 FPS 的估算边界，
40k文本/25图/50k完整condition是留显存余量的准入建议，尚未作为统一服务限制实施，不能直接套用当前默认布局。

## 验证边界与回退

本次默认值调整不代表完成三项优化组合的 GPU 端到端验收，也不叠乘独立 PR 收益。
原验证范围及画质状态见 [HISTORY_ROPE_GROUPED_PLAN.md](https://github.com/seedleap/sglang/blob/main/benchmark/zing_h3_causal/HISTORY_ROPE_GROUPED_PLAN.md) 和 [CHUNK_ROPE_REUSE.md](CHUNK_ROPE_REUSE.md)。
需要旧路径时显式设置前三项为 `0`；plan 保持 `0`。

## grouped 与 chunk reuse 的组合（#211）

默认 chunk reuse 路径现在将 `rotate_batch` 传入失效片段重建：每个连续失效区间
调用一次 grouped，中间已命中的片段既不重算也不覆盖，全命中时不调用历史旋转。
不为合批拼接整个历史 K；原始版本、坐标和物理输出范围的精确复用判定保持不变。
所有历史写入和 current-tail 写入成功后才发布本层结果，异常后下一次完整重建。

- `GROUPED_HISTORY_ROPE=0`：保留 chunk reuse，失效片段逐片段旋转。
- `CHUNK_ROPE_REUSE=0`：恢复原先 common-prefix 重用及其 grouped 后缀路径。
- `FUSED_HISTORY_ROPE=0`：grouped 分发仍可发生，计算回退 eager；不支持的设备、
  dtype、布局和 alias 同样沿用已有回退。
- `GROUPED_HISTORY_ROPE=contiguous`：仍为显式替代模式。仅在当前连续失效区间的
  raw K 可作为邻接 view 时打包系数并调用 fused，否则走既有 grouped/fallback。
- `HISTORY_ROPE_PLAN` 仍默认 `0`。启用后保留 owner/state/positions 校验和 history
  selection；chunk reuse 的有效性与区间分组仍由已绑定的 chunk plan 决定，
  不使用 history plan 的 common-prefix geometry 替代逐片段有效性判定。

针对性复现命令和实际证据见 [GROUPED_CHUNK_REUSE.md](https://github.com/seedleap/sglang/blob/main/benchmark/zing_h3_causal/GROUPED_CHUNK_REUSE.md)。

## 480p 固定槽路径

默认优化只覆盖 video-only、SP2、56 attention heads / head_dim128、CPU 元数据
确认的 15×26 空间网格（832×480 的 checkpoint patch 网格）、已测 query 行数，
以及 FA4 `4.0.0b19`、CUDA CC10.3 / 148 SM。前缀、其他网格、硬件与后端保留
原路径；显式开启 RoPE/KV 融合仍受 dtype、布局和索引范围检查约束。
单独设置 `SGLANG_ZING_FUSED_CURRENT_ROPE_KV=0` 可关闭当前融合而保留最佳分段 FA。
另设 `SGLANG_ZING_FA4_FIXED=0` 可对照原 varlen 路由；开关参与 Graph key，
不会误用不同路由捕获的 Graph。此路径不采用联合 sparse/flex attention。

融合保持 BF16 乘法后舍入再相加的顺序。当前旋转 K/V 写入槽尾用于本次
attention，只有 clean rows 的 raw K 持久化；历史坐标、窗口淘汰和提交规则不变。
GPU KV 有效长度仍直接传给 FA，不把固定容量当成历史长度，不在 replay 时读回 CPU。

模型签名仍逐次检查参数及 buffer 的 identity、地址、版本、shape、stride、dtype
和 device；只复用未变的不可变签名及其 hash。共享 KV chunk 的层视图按需创建，
避免计划重建时生成完整逐层 Tensor 视图。GC 策略不变。单次性能数据和验收记录见 PR，
不将某轮 TTFF 中位数下降视为已证明稳定改善。

## 736p FPS 更新（2026-10-30）

本节只更新性能记录，不修改默认开关。历史 1312×736/SP4/W21 的 8.3464 FPS 不能
作为当前部署指标。以下输出均为 **1280×736**，FPS 为中位数：

| 硬件 / 配方 | source | 全程 FPS | 首包后 FPS | 正式 n |
| --- | --- | ---: | ---: | ---: |
| B300，SP4/resident、BF16、4→2，产品少图 | 1280×736 | 18.411 | 19.951 | 3 |
| 同上，长 prompt | 同上 | **20.589** | 20.830 | 3 |
| 同上，动态换图 | 同上 | **20.386** | 21.283 | 3 |
| 同上，长会话 | 同上 | **20.626** | 20.669 | 2 / 计划3 |
| B300，SP2，固定废土，`4c5e387948` | 1280×736 | 19.358 | 20.109 | 3 |
| H200，SP2、BF16 #401/#406、4→2，固定废土 | 1280×736 | **8.486** | 8.880 | 3 |
| H200，SP2、BF16、4→2，固定废土 | 832×480 | **12.783** | 13.327 | 3 |
| H200，SP2、BF16、4→1，固定废土 | 832×480 | **16.357** | 17.273 | 3 |

B300 产品结果冻结于 `521ade5a6e`、旧 split-first 模型，实际六卡、分配八卡；
W6/S3、Graph、quantum1、Encoder/Decoder overlap、cross-chunk joint、预热 Block0 cache，
无 offload、无 FP8 GEMM。11/12 次正式完成后作业被外部停止，长会话第三轮排除，
SP2 对照和 DiT 诊断未执行。最新 main、新模型、所有 case 稳定20 FPS及完整效果仍未验证。
H200 FP8 组合另有三个旧 case 全程 9.897–10.493 FPS，不能视为 BF16 默认或新产品结果。
不同硬件、case、source 尺寸及采样不作收益比较；一步/source480 的质量须单独验收。
完整配置、版本、P95 和原始证据入口见
[736p 部署性能更新](../deployment_zing_h3.mdx#736p-部署性能更新2026-10-30)。

## VAE 执行默认值（2026-11-05 / 2026-11-12 更新）

按用户指定，`MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT`、`MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT`、`SGLANG_LYNN_PACKED_QKV`、`SGLANG_LYNN_FUSED_QK_NORM`、`SGLANG_LYNN_COMPILED_FFN` 五项默认 `1`；`SGLANG_LYNN_SM90_FAST_DECODE` 于 2026-11-13 按用户确认默认 `1`；完整 product-case 冻结 latent Decoder A/B 已完成，三角色 E2E 与人工画质仍待验收；旧版 workspace、整块 compile 的选项已移除。显式环境变量优先，已有部署需移除旧的关闭覆盖并重启对应 worker。Lynn 显式传入 `execution_options` 时沿用调用者配置；`SGLANG_LYNN_SM90_FAST_DECODE=1` 仅在 SM90（CC 9.0）单卡（`world_size == 1`）且输入满足架构准入条件时激活，在非 SM90、多卡 SP、CPU 或小尺寸输入上自动无感回退块内融合或参考路径。

这是运行默认策略，不是严格无损或端到端整体 FPS 提升承诺。2026-11-12 已在上海 H100（`NVIDIA L20Z`, SM90, `1c0246cfec`）完成单卡 Lynn Decoder 组件验收（`832×480` 稳态单步中位 `9.759 ms` / P95 `9.824 ms`；`1280×736` 稳态单步中位 `22.361 ms` / P95 `23.099 ms`），完整三角色在线流 E2E、H200/B300 及精度/硬件边界见 [VAE 执行方案](../plans/VAE_EXECUTION_OPTIMIZATIONS.md)。
