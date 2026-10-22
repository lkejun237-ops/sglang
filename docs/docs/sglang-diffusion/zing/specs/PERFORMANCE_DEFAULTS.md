# Zing 性能优化默认配置

基于主线 `df69ad4265` 梳理最近合入的优化。下表环境变量均省略
`SGLANG_ZING_` 前缀（单独注明者除外）。显式设置仍优先于默认值。

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
