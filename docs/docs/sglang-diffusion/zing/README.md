# Zing 设计文档

架构、接口契约与设计说明保存在 `specs/`，实现计划与设计分析保存在 `plans/`。
迁入文档的历史配置、冻结提交和验收边界保持原样，不据此扩大当前支持范围。

| 主题 | 文档 |
| --- | --- |
| 增量编解码与缓存 | [Incremental codecs](specs/INCREMENTAL_CODECS.md) |
| 渐进 source encoder | [接入契约](specs/PROGRESSIVE_ENCODER.md)、[tile 并行设计](specs/PROGRESSIVE_TILE_PARALLEL_DESIGN.md) |
| Encoder / DiT 流水线 | [Encoder overlap](specs/ENCODER_OVERLAP.md)、[跨 dispatch 流水线](specs/CROSS_DISPATCH_PIPELINE.md) |
| Source 与参考图输入 | [输入契约](specs/SOURCE_REFERENCE_INPUTS.md)、[有界传输](specs/SOURCE_TRANSPORT.md) |
| 联合 forward 与去噪调度 | [Joint forward](specs/JOINT_FORWARD.md)、[初始分块 schedule](specs/INITIAL_BLOCK_SCHEDULE.md) |
| RoPE 与性能配置 | [Chunk RoPE reuse](specs/CHUNK_ROPE_REUSE.md)、[性能默认配置](specs/PERFORMANCE_DEFAULTS.md) |
| AdaLN 导出与离线表 | [Split export](specs/SPLIT_ADALN_EXPORT.md)、[AdaLN table](specs/ADALN_TABLE.md)、[设计分析](plans/ISSUE_257_ANALYSIS.md) |
| Conditioning 服务 | [RPC 使用与部署契约](specs/ISSUE_256_CONDITIONING_USAGE.md) |

其他设计与契约见 [specs/](specs/)。运行时内部流式设计见
[STREAMING_DESIGN.md](../../../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/STREAMING_DESIGN.md)。
测试输入、复现工具和验收记录保留在 [Zing benchmark](../../../../benchmark/zing_h3_causal/README.md)。
