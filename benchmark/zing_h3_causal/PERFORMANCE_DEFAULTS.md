# Zing 性能优化默认配置

基于主线 `df69ad4265` 梳理最近合入的优化。下表环境变量均省略
`SGLANG_ZING_` 前缀（单独注明者除外）。显式设置仍优先于默认值。

| 字段 | 默认值 | 用途 |
| --- | --- | --- |
| `FUSED_HISTORY_ROPE` | `1`（本次开启） | 融合历史 K RoPE；不支持的设备和布局仍回退 |
| `FUSED_CURRENT_ROPE` | `1` | 当前块 Q/K RoPE 复用融合历史 RoPE kernel，与 eager 逐位一致；不支持的设备和布局（含 SP）回退 |
| `GROUPED_HISTORY_ROPE` | `1`（本次开启） | 多片段 grouped 处理；`contiguous` 是替代模式 |
| `CHUNK_ROPE_REUSE` | `1`（本次开启） | 按 chunk 复用未失效的历史 RoPE |
| `HISTORY_ROPE_PLAN` | `0` | 跨层复用 forward metadata；现有实测无稳定额外收益 |
| `EXPERIMENTAL_ATTENTION` | `fa4` | 已默认使用 FA4 |
| `EXPERIMENTAL_STABLE_SHAPES` | `1` | 已开启稳定形状 |
| `EXPERIMENTAL_COMPILED_ADALN` | `1` | 已开启编译 AdaLN |
| `EXPERIMENTAL_FA4_METADATA_REUSE` | `1` | 已开启 forward 内 FA4 长度元数据复用 |
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
原验证范围及画质状态见 HISTORY_ROPE_GROUPED_PLAN.md 和 CHUNK_ROPE_REUSE.md。
需要旧路径时显式设置前三项为 `0`；plan 保持 `0`。
