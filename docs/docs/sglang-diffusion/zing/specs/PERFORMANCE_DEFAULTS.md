# Zing 性能优化默认配置

基于主线 `df69ad4265` 梳理最近合入的优化。下表环境变量均省略
`SGLANG_ZING_` 前缀（单独注明者除外）。显式设置仍优先于默认值。

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

这组是部署容量契约，不能在未知显存和并发需求时任意填值：

| 完整字段 | 默认值 |
| --- | --- |
| `zing_kv_pool_size` | `None`（未配置池） |
| `zing_kv_pool_max_target_area` | `None` |
| `zing_kv_pool_max_target_long_edge` | `None` |
| `zing_kv_pool_max_prefix_rows` | `None` |
| `zing_kv_pool_window_chunks` | `21` |
| `zing_kv_pool_sink_chunks` | `3` |
| `zing_kv_pool_rope_position_mode` | `block_relative` |
| `zing_kv_pool_rope_max_frame_gap` | `12` |

当前请求默认 window21 / sink3、block_relative、max_frame_gap=12；
解码模式仍为 `stable`。修改窗口或改为 stateful 解码会改变请求语义，不属于纯算子开关。

选择 prompt/reference 预算时，可参考 [H200 两卡输入预算](/docs/sglang-diffusion/zing/prompt-reference-budgets)。
该页基于历史 W6/S3、拆分首块、Graph 关闭的 DiT 实测；25.6k 是约20 FPS 的估算边界，
40k文本/25图/50k完整condition是留显存余量的准入建议，尚未作为统一服务限制实施，不能直接套用当前默认布局。

## 验证边界与回退

本次默认值调整不代表完成三项优化组合的 GPU 端到端验收，也不叠乘独立 PR 收益。
原验证范围及画质状态见 [HISTORY_ROPE_GROUPED_PLAN.md](../../../../../benchmark/zing_h3_causal/HISTORY_ROPE_GROUPED_PLAN.md) 和 [CHUNK_ROPE_REUSE.md](CHUNK_ROPE_REUSE.md)。
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

针对性复现命令和实际证据见 [GROUPED_CHUNK_REUSE.md](../../../../../benchmark/zing_h3_causal/GROUPED_CHUNK_REUSE.md)。

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
